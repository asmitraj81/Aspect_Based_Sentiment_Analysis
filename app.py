#@title K-Tran ABSA: Complete Training and Testing (Direct File Upload)
    attention_mask = encoded['attention_mask'].to(device)

    doc = nlp_demo(sentence)
    matrix = torch.zeros(max_seq_len, max_seq_len, dtype=torch.float32)
    offset_mapping = encoded.offset_mapping.squeeze(0).tolist()
    for token in doc:
        char_start, char_end = token.idx, token.idx + len(token.text)
        token_indices = [i for i, (start, end) in enumerate(offset_mapping) if start < char_end and end > char_start]
        if not token_indices: continue
        head_char_start, head_char_end = token.head.idx, token.head.idx + len(token.head.text)
        head_indices = [i for i, (start, end) in enumerate(offset_mapping) if start < head_char_end and end > head_char_start]
        if head_indices:
            for i in token_indices:
                for j in head_indices:
                    matrix[i][j] = 1
                    matrix[j][i] = 1
    for i in range(max_seq_len): matrix[i][i] = 1
    syntax_matrix = matrix.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = final_model(input_ids, attention_mask, syntax_matrix=syntax_matrix)

    ate_preds = torch.argmax(outputs['ate_logits'], dim=2).squeeze(0)
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))

    aspects = []
    current_aspect_tokens = []
    for i, pred in enumerate(ate_preds):
        if not attention_mask[0, i] or tokens[i] in (tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token):
            continue

        token_id = tokenizer.convert_tokens_to_ids(tokens[i])
        if pred == 1: # B-Aspect
            if current_aspect_tokens: aspects.append(tokenizer.decode(current_aspect_tokens))
            current_aspect_tokens = [token_id]
        elif pred == 2 and current_aspect_tokens: # I-Aspect
            current_aspect_tokens.append(token_id)
        else: # O-token
            if current_aspect_tokens:
                aspects.append(tokenizer.decode(current_aspect_tokens))
                current_aspect_tokens = []
    if current_aspect_tokens: aspects.append(tokenizer.decode(current_aspect_tokens))

    sentiment_pred = torch.argmax(outputs['sentiment_logits'], dim=1).item()
    sentiment_map_inv = {v: k for k, v in config['data']['sentiment_map'].items()}
    sentiment = sentiment_map_inv.get(sentiment_pred, "unknown")

    if not aspects:
        return "No aspects detected.", {}

    results = {}
    for aspect in aspects:
        results[aspect.strip()] = sentiment.upper()

    return f"Overall Sentiment: {sentiment.upper()}", results

# Launch Gradio Interface
interface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, label="Enter a sentence", placeholder="The food was amazing but the service was slow."),
    outputs=[gr.Textbox(label="Overall Sentiment"), gr.Label(label="Aspects and Sentiments")],
    title="🧪 K-Tran ABSA: Interactive Demo",
    description="Test the K-Tran Syntax-Aware Model. This model identifies aspects in a sentence and classifies their sentiment.",
    examples=[
        ["The service is excellent but the food is terrible."],
        ["I loved the ambiance, and the pasta was cooked perfectly."],
        ["The sushi was incredibly fresh and the presentation was beautiful."]
    ]
)
interface.launch(debug=True)