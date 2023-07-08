import os, sys, torch
from transformers import AutoConfig, AutoTokenizer, BatchEncoding, OPTModel
import time

# "facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b",
# "facebook/opt-2.7b", "facebook/opt-6.7b", "facebook/opt-13b",
# "facebook/opt-30b", "facebook/opt-66b"


def generate_stories(fn):
    stories = []
    f = open(fn)
    first_line = f.readline()
    assert first_line.strip() == "!ARTICLE"
    curr_story = ""

    for line in f:
        sentence = line.strip()
        if sentence == "!ARTICLE":
            stories.append(curr_story[:-1])
            curr_story = ""
        else:
            curr_story += line.strip() + " "

    stories.append(curr_story[:-1])
    return stories


def main():
    corpus = sys.argv[1].split("/")[-1].split(".")[0]
    stories = generate_stories(sys.argv[1])

    model_variant = sys.argv[2].split("/")[-1]
    dec_len = int(sys.argv[3])

    config = AutoConfig.from_pretrained(sys.argv[2])
    config.update({"dec_len": dec_len})

    tokenizer = AutoTokenizer.from_pretrained(sys.argv[2], use_fast=False)
    model = OPTModel.from_pretrained(sys.argv[2], config=config)
    model.eval()
    ctx_size = model.config.max_position_embeddings
    bos_id = model.config.bos_token_id

    try:
        os.makedirs("output")
    except FileExistsError:
        pass

    if not os.path.exists(f"output/{model_variant}_proj_weights.pt"):
        torch.save(model.decoder.embed_tokens.weight, f"output/{model_variant}_proj_weights.pt")

    batches = []
    for story in stories:
        # OPT tokenizer automatically adds <s> to beginning
        tokenizer_output = tokenizer(story)
        ids = tokenizer_output.input_ids[1:]
        attn = tokenizer_output.attention_mask[1:]
        start_idx = 0

        # in case the text doesn't fit in one context window
        while len(ids) > ctx_size:
            batches.append((BatchEncoding(
                {"input_ids": torch.tensor([bos_id] + ids[:ctx_size-1]).unsqueeze(0),
                 "attention_mask": torch.tensor([1] + attn[:ctx_size-1]).unsqueeze(0)}), start_idx))
            ids = ids[int(ctx_size/2):]
            attn = attn[int(ctx_size/2):]
            start_idx = int(ctx_size/2)-1

        # remaining tokens
        batches.append((BatchEncoding(
            {"input_ids": torch.tensor([bos_id] + ids).unsqueeze(0),
             "attention_mask": torch.tensor([1] + attn).unsqueeze(0)}), start_idx))

    for i, batch in enumerate(batches):
        batch_input, start_idx = batch
        identifier = f"{corpus}_{model_variant}_{i}"
        batch_input["identifier"] = identifier

        with torch.no_grad():
            torch.save(batch_input.input_ids, f"output/i_{identifier}.pt")
            print(f"Started batch {i} of length {batch_input.input_ids.shape[1]} at", time.strftime("%H:%M:%S", time.localtime()))
            _ = model(**batch_input)
            print(f"Finished batch {i} of length {batch_input.input_ids.shape[1]} at", time.strftime("%H:%M:%S", time.localtime()))


if __name__ == "__main__":
    main()
