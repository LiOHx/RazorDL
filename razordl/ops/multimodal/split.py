"""Split multimodal batch tensors into micro-batches."""


def split_multi_modal_input_dict(input_dict, processor, batch_size=1):
    """Split a processor output dict into micro-batches.

    Handles text, image, and video fields automatically by tracking
    vision tokens via ``<|vision_start|>`` and matching ``grid_thw``
    entries to samples.
    """
    total_samples = input_dict["input_ids"].size(0)
    if total_samples < batch_size:
        return [input_dict]

    has_image = "pixel_values" in input_dict and input_dict["pixel_values"] is not None
    has_video = "pixel_values_videos" in input_dict and input_dict["pixel_values_videos"] is not None

    if has_image or has_video:
        vision_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        images_per_sample = [
            (input_dict["input_ids"][i] == vision_start_token_id).sum().item()
            for i in range(total_samples)
        ]

        grid_thw_key = "video_grid_thw" if has_video else "image_grid_thw"

        pixels_per_sample = []
        img_idx = 0
        for num_images in images_per_sample:
            if num_images > 0:
                sample_grid_thw = input_dict[grid_thw_key][img_idx:img_idx + num_images]
                num_pixels = int(sample_grid_thw.prod(dim=1).sum().item())
            else:
                num_pixels = 0
            pixels_per_sample.append(num_pixels)
            img_idx += num_images

    result_lst = []
    image_idx = 0
    pixel_idx = 0

    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)

        batch_dict = {
            "input_ids": input_dict["input_ids"][start_idx:end_idx],
            "attention_mask": input_dict["attention_mask"][start_idx:end_idx],
        }

        if has_video:
            batch_num_images = sum(images_per_sample[start_idx:end_idx])
            batch_num_pixels = sum(pixels_per_sample[start_idx:end_idx])
            batch_dict["pixel_values_videos"] = input_dict["pixel_values_videos"][pixel_idx:pixel_idx + batch_num_pixels]
            batch_dict["video_grid_thw"] = input_dict["video_grid_thw"][image_idx:image_idx + batch_num_images]
            image_idx += batch_num_images
            pixel_idx += batch_num_pixels

        if has_image:
            batch_num_images = sum(images_per_sample[start_idx:end_idx])
            batch_num_pixels = sum(pixels_per_sample[start_idx:end_idx])
            batch_dict["pixel_values"] = input_dict["pixel_values"][pixel_idx:pixel_idx + batch_num_pixels]
            batch_dict["image_grid_thw"] = input_dict["image_grid_thw"][image_idx:image_idx + batch_num_images]
            image_idx += batch_num_images
            pixel_idx += batch_num_pixels

        excluded_keys = {
            "input_ids", "attention_mask", "pixel_values", "image_grid_thw",
            "pixel_values_videos", "video_grid_thw",
        }
        for key in input_dict.keys():
            if key not in excluded_keys:
                batch_dict[key] = input_dict[key][start_idx:end_idx]

        result_lst.append(batch_dict)

    return result_lst
