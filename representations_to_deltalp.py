import sys, torch
from transformers import AutoTokenizer


def main():
    identifier = sys.argv[1]
    tokenizer = AutoTokenizer.from_pretrained(sys.argv[2])
    model_variant = sys.argv[2].split("/")[-1]

    p = torch.load(f"output/{model_variant}_proj_weights.pt")  # vocab_size, embed_dim
    i = torch.load(f"output/i_{identifier}.pt").squeeze(0)  # tgt_len
    d = torch.load(f"output/d_{identifier}.pt")  # dec_len, tgt_len, embed_dim
    o = torch.load(f"output/o_{identifier}.pt").squeeze(0)  # tgt_len, embed_dim

    dec_len, tgt_len, _ = d.shape

    softmax = torch.nn.Softmax(dim=-1)
    tokens = tokenizer.convert_ids_to_tokens(i)

    distances = []

    for j in range(tgt_len-1):

        stride_len = min(j+1, dec_len)

        final_logit = o[j].matmul(p.t())  # vocab_size
        final_probs = softmax(final_logit).detach()  # vocab_size
        final_lp = torch.log2(final_probs[i[j+1]])

        dec_logit = torch.flipud(d[:stride_len,j].matmul(p.t()))  # stride_len, vocab_size

        print(f"\n{identifier} Timestep {j+1}/{tgt_len-1}")

        context = " ".join(tokens[:j+1])
        pred_token = tokens[j+1]

        print(f"LP({pred_token} | {context}) = {final_lp}")
        best = 0
        argmax = -1

        for k in range(stride_len):
            ablated_logit = final_logit - dec_logit[k]
            ablated_probs = softmax(ablated_logit).detach()

            ablated_token = tokens[j-stride_len+k+1]
            ablated_lp = torch.log2(ablated_probs[i[j+1]])

            if (j-stride_len+k+1 != 0) and (ablated_lp < best):
                argmax = j-stride_len+k+1
                best = ablated_lp

            print(f"-{ablated_token} = {ablated_lp};\tDelta LP: {final_lp-ablated_lp}")

        print(f"Timestep {j+1}, argmax: {argmax}, distance: {j-argmax+1}")
        distances.append(j-argmax+1)

    max_distance = max(distances)
    print(f"Max distance: {max_distance}")


if __name__ == "__main__":
    main()



