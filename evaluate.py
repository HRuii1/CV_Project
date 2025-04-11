# video-captioning/evaluate.py

import os
import torch
from tqdm import tqdm
from config import Config
from data.dataloader import get_pt_dataloader, read_list_file
from data_processing_augmented import read_captions_file, read_list_file
from transformers import GPT2Tokenizer
from models.c3d_model import C3DEncoder
from models.clip_model import CLIPEncoder
from models.fusion import SimpleFusion
from models.gpt2_decoder import GPT2Decoder
from utils.metrics import calculate_bleu
from utils.beam_search import beam_search_decoding

# Example dummy annotations (video_id, ref_caption)
train_ids = read_list_file(Config.TRAIN_LIST)
caption_dict = read_captions_file(Config.CAPTIONS_FILE)
VAL_ANNOTATIONS = [(vid, cap) for vid in train_ids for cap in caption_dict.get(vid, [])]


def generate_caption_c3d(c3d_encoder, fusion_model, gpt2_decoder, video_feats, tokenizer, device, beam_width=1):
    # video_feats shape: (1, 16, 3, 112, 112)
    video_feats = video_feats.permute(0, 2, 1, 3, 4)
    with torch.no_grad():
        enc_out = c3d_encoder(video_feats)  # (1, 15360)
        context_embeds = fusion_model(enc_out)  # (1, T_ctx, 768)
    
    if beam_width > 1:
        # custom beam search
        caption_str = beam_search_decoding(
            gpt2_decoder,
            context_embeds,
            beam_width=beam_width,
            max_steps=Config.MAX_SEQ_LEN,
            tokenizer=tokenizer
        )
    else:
        # use GPT-2 generate with greedy
        gen_ids = gpt2_decoder.generate(
            context_embeds,
            max_length=Config.MAX_SEQ_LEN,
            num_beams=1
        )
        caption_str = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    return caption_str

def generate_caption_clip(clip_encoder, fusion_model, gpt2_decoder, video_feats, tokenizer, device, beam_width=1):
    # video_feats shape: (1, F, 512)
    with torch.no_grad():
        enc_out = clip_encoder(video_feats)        # (1, F, 768)
        context_embeds = fusion_model(enc_out)     # (1, T_ctx, 768)
    
    if beam_width > 1:
        caption_str = beam_search_decoding(
            gpt2_decoder,
            context_embeds,
            beam_width=beam_width,
            max_steps=Config.MAX_SEQ_LEN,
            tokenizer=tokenizer
        )
    else:
        gen_ids = gpt2_decoder.generate(
            context_embeds,
            max_length=Config.MAX_SEQ_LEN,
            num_beams=1
        )
        caption_str = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    return caption_str

def main():
    device = torch.device(Config.DEVICE)
    tokenizer = GPT2Tokenizer.from_pretrained(Config.GPT2_MODEL_NAME)
    
    # Load your best checkpoint
    checkpoint_path = os.path.join(Config.OUTPUT_DIR, "model_epoch_2.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model instances
    c3d_encoder = None
    clip_encoder = None
    # set to True or False depending on which you used
    USE_C3D = True
    
    if USE_C3D:
        c3d_encoder = C3DEncoder().to(device)
        c3d_encoder.load_state_dict(checkpoint["c3d_encoder"])
    else:
        clip_encoder = CLIPEncoder(freeze_clip=Config.FREEZE_CLIP).to(device)
        clip_encoder.load_state_dict(checkpoint["clip_encoder"])
    
    fusion_model = SimpleFusion(context_tokens=Config.CONTEXT_TOKENS).to(device)
    fusion_model.load_state_dict(checkpoint["fusion_model"])
    
    gpt2_decoder = GPT2Decoder().to(device)
    gpt2_decoder.load_state_dict(checkpoint["gpt2_decoder"])
    
    # Prepare dataset/loader for evaluation
    # val_loader = get_pt_dataloader(
    #     feature_pt_path=Config.PROCESSED_C3D_FEATS if USE_C3D else Config.PROCESSED_CLIP_FEATS,
    #     split_name="test",  # this accesses version1["test"]
    #     annotations=test_annotations,
    #     tokenizer=tokenizer,
    #     batch_size=1,
    #     shuffle=False
    # )

    # for simplicity, we will use the same train annotations for validation
    val_loader = get_pt_dataloader(
        feature_pt_path=Config.PROCESSED_C3D_FEATS,  # or Config.PROCESSED_CLIP_FEATS
        split_name="train",  # not "test"
        annotations=VAL_ANNOTATIONS,
        tokenizer=tokenizer,
        batch_size=1,
        shuffle=False
    )
    
    references = []
    hypotheses = []
    
    for batch in tqdm(val_loader, desc="Evaluating"):
        video_feats, in_ids, att_msks = batch
        video_feats = video_feats.to(device)
        
        # just take the reference text from in_ids for BLEU
        ref_str = tokenizer.decode(in_ids[0], skip_special_tokens=True)
        references.append(ref_str.split())
        
        if c3d_encoder:
            hyp_str = generate_caption_c3d(
                c3d_encoder, 
                fusion_model, 
                gpt2_decoder, 
                video_feats, 
                tokenizer,
                device,
                beam_width=5  # or 1 for greedy
            )
        else:
            hyp_str = generate_caption_clip(
                clip_encoder,
                fusion_model,
                gpt2_decoder,
                video_feats,
                tokenizer,
                device,
                beam_width=5
            )
        
        hypotheses.append(hyp_str.split())
    
    # Compute average BLEU
    bleu_scores = []
    for ref, hyp in zip(references, hypotheses):
        bleu = calculate_bleu(ref, hyp)
        bleu_scores.append(bleu)
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f"Validation BLEU-4: {avg_bleu:.4f}")

if __name__ == "__main__":
    main()
