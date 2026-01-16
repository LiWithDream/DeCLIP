import torch
import torch.nn as nn
import torch.nn.functional as F
from training.misc import is_main_process
from training.dcac_loss import compute_dcac_loss


class DeCLIP:
    def _encode_patch_tokens(self, model, images):
        visual = model.visual if hasattr(model, "visual") else model.vision_model
        patch_dropout = getattr(visual, "patch_dropout", None)
        if patch_dropout is not None and not isinstance(patch_dropout, nn.Identity):
            visual.patch_dropout = nn.Identity()
            try:
                tokens = visual(images, return_all_features=True)
            except TypeError:
                tokens = visual.forward(images, return_all_features=True)
            finally:
                visual.patch_dropout = patch_dropout
        else:
            try:
                tokens = visual(images, return_all_features=True)
            except TypeError:
                tokens = visual.forward(images, return_all_features=True)
        if tokens.dim() != 3:
            raise ValueError("Expected token sequence output for DCAC features.")
        tokens = tokens[:, 1:, :]
        if hasattr(visual, "norm"):
            tokens = visual.norm(tokens)
        return tokens

    def __call__(self, batch, student, teacher, vfm_model, args):
        losses={}
        context_weight = args.loss_context_weight
        content_weight = args.loss_content_weight
        if args.distributed:
            student = student.module
        dtype_map = {"bf16": torch.bfloat16, "amp": torch.float16}
        input_dtype = dtype_map.get(args.precision, torch.float32)
        images_view1 = None
        images_view2 = None
        overlap_meta = None
        if len(batch) == 4:
            images, normed_boxes, image_crops, proxy_image = batch
        elif len(batch) == 5:
            images, normed_boxes, image_crops, proxy_image, _ = batch
        elif len(batch) == 7:
            images, normed_boxes, image_crops, proxy_image, images_view1, images_view2, overlap_meta = batch
        elif len(batch) == 8:
            images, normed_boxes, image_crops, proxy_image, images_view1, images_view2, overlap_meta, _ = batch
        else:
            raise ValueError(f"Unexpected batch size for DeCLIP: {len(batch)}")
        images = images.to(device=args.device, dtype=input_dtype, non_blocking=True)  
        normed_boxes = normed_boxes.to(device=args.device, dtype=input_dtype,non_blocking=True)
        image_crops = image_crops.to(device=args.device, dtype=input_dtype,non_blocking=True) 
        proxy_image=proxy_image.to(device=args.device, dtype=input_dtype, non_blocking=True)
        if args.use_dcac and images_view1 is not None:
            images_view1 = images_view1.to(device=args.device, dtype=input_dtype, non_blocking=True)
            images_view2 = images_view2.to(device=args.device, dtype=input_dtype, non_blocking=True)
            overlap_meta = {k: v.to(device=args.device, non_blocking=True) for k, v in overlap_meta.items()}

        rois_list = []
        crops_list = []
        for bboxes_per_image, crops_per_image in zip(normed_boxes, image_crops):
            valid = bboxes_per_image[:, -1] > 0.5
            rois_list.append(bboxes_per_image[valid, :4])
            crops_list.append(crops_per_image[valid])
        image_crops = torch.cat(crops_list)
        student_roi_features, context =student.encode_pseudo_boxes(images, rois_list, normalize=True, mode = args.mode)

        with torch.no_grad():
            teacher_crop_features = teacher.encode_image(image_crops, normalize=True)
            if args.use_vfm:
                teacher_context_similarity, teacher_h, teacher_w = self.get_teacher_context_similarity(vfm_model,proxy_image,args)
            else:
                teacher_context_similarity=teacher_h = teacher_w=None

        if args.use_vfm:
            student_context_similarity=self.get_student_context_similarity(images, context,teacher_h,teacher_w, args)
            _loss_context=(teacher_context_similarity - student_context_similarity).norm(p=2,dim=-1).mean() 
            losses.update({"loss_context":_loss_context*context_weight})

        _loss_content = 1.0 - (student_roi_features * teacher_crop_features).sum(-1).mean()
        losses.update({"loss_content":_loss_content * content_weight})

        if args.use_dcac and images_view1 is not None:
            patch_feat1 = self._encode_patch_tokens(student, images_view1)
            patch_feat2 = self._encode_patch_tokens(student, images_view2)
            loss_dcac = compute_dcac_loss(patch_feat1, patch_feat2, overlap_meta, temp=args.dcac_temp)
            losses.update({"loss_dcac": loss_dcac * args.dcac_weight})
        return losses, len(images)

    def get_teacher_context_similarity(self,vfm_model,proxy_image,args):
        if 'sam' in args.use_vfm:
            vfm_feats=vfm_model.image_encoder(proxy_image)
        elif "dinov2" in args.use_vfm:
            vfm_feats = vfm_model.get_intermediate_layers(proxy_image, reshape=True)[0]
        elif 'dino' in args.use_vfm:
            feat = vfm_model.get_intermediate_layers(proxy_image)[0]
            nb_im = feat.shape[0]
            patch_size = vfm_model.patch_embed.patch_size
            I, J = proxy_image[0].shape[-2] // patch_size, proxy_image[0].shape[-2] // patch_size
            vfm_feats = feat[:, 1:, :].reshape(nb_im, I, J, -1).permute(0, 3, 1, 2)
        else:
                raise NotImplementedError(f"mode {args.use_vfm} is not implemented yet.")
        teacher_h,teacher_w=vfm_feats.shape[-2:]
        vfm_feats= F.normalize(vfm_feats.flatten(-2,-1), dim=1)
        vfm_similarity = torch.einsum("b c m, b c n -> b m n", vfm_feats, vfm_feats)
        return vfm_similarity,teacher_h,teacher_w
    
    def get_student_context_similarity(self,images,context,teacher_h,teacher_w,args):
        B = images.shape[0]
        if args.mode in ["qq_vfm_distill","kk_vfm_distill","vv_vfm_distill","sanity_check"]:
            N, _ = context.shape[1:]
            context=context.transpose(0, 1).contiguous().view(N, B, -1).transpose(0, 1)
            bs, N, C = context.shape
            n_sqrt = int(N ** 0.5)
            if n_sqrt != teacher_h or n_sqrt != teacher_w:
                context_reshaped = context.transpose(-2,-1).contiguous().view(bs, C, n_sqrt, n_sqrt)
                context_resized = F.interpolate(context_reshaped, size=(teacher_h, teacher_w), mode='bilinear', align_corners=False)
                context = context_resized.transpose(-2,-1).contiguous().view(bs, teacher_h * teacher_w, C)
            context = F.normalize(context, dim=-1).transpose(-2,-1)
            student_context_similarity=torch.einsum("b c m, b c n -> b m n", context, context)
        elif args.mode == "csa_vfm_distill":
            q_feature, k_feature = context
            N, _ = q_feature.shape[1:]
            q_feature = q_feature.transpose(0, 1).contiguous().view(N, B, -1).transpose(0, 1)
            k_feature = k_feature.transpose(0, 1).contiguous().view(N, B, -1).transpose(0, 1)
            q_feature = F.normalize(q_feature, dim=-1).transpose(-2,-1)
            k_feature = F.normalize(k_feature, dim=-1).transpose(-2,-1)
            student_context_similarity=(torch.einsum("b c m, b c n -> b m n", q_feature, q_feature)+torch.einsum("b c m, b c n -> b m n", k_feature, k_feature))/2.0
        elif args.mode == "all_vfm_distill": 
            q_feature, k_feature, v_feature = context
            q_feature = F.normalize(q_feature, dim=-1).transpose(-2,-1)
            k_feature = F.normalize(k_feature, dim=-1).transpose(-2,-1)
            v_feature = F.normalize(v_feature, dim=-1).transpose(-2,-1)
            student_context_similarity=(torch.einsum("b c m, b c n -> b m n", q_feature, q_feature)+
                                    torch.einsum("b c m, b c n -> b m n", k_feature, k_feature)+
                                    torch.einsum("b c m, b c n -> b m n", v_feature, v_feature))/3.0
           
        else:
            raise NotImplementedError(f"Mode '{args.mode}' is not implemented.")
        return student_context_similarity
