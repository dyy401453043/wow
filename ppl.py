    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=10,
        collate_fn=collator
    )
    model.eval()
    total = 0
    exactmatch = []
    hyp, refs = [], []
    test_losses = []
    count_batch = 0
    model = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, labels, _, context_ids, context_mask) = batch

            test_loss = model(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                labels=labels.cuda()
            )[0]
            test_losses.append(test_loss.item())
            count_batch += 1

            outputs = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                max_length=50
            )

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                hyp.append(ans)
                gold = dataset.get_example(idx[k])['answers']
                refs.append(gold)
                score = src.evaluation.ems(ans, gold)
                total += 1
                exactmatch.append(score)

    exactmatch, total = src.util.weighted_average(np.mean(exactmatch), total, opt)
    test_loss, count_batch = src.util.weighted_average(np.mean(test_losses), count_batch, opt)
    ppl = np.exp(test_loss)
    sacrebleu_score = round(metric_sacrebleu.compute(predictions=hyp, references=refs)["score"], 4)
    meteor_score = round(metric_meteor.compute(predictions=hyp, references=[ref[0] for ref in refs])["meteor"] * 100, 4)
    rouge_score = round(metric_rouge.compute(predictions=hyp, references=[ref[0] for ref in refs])["rougeL"].mid.fmeasure * 100, 4)
    f1 = round(sum([f1_score(i, j) for i, j in zip(hyp, [ref[0] for ref in refs])])/len(hyp) * 100, 4)

    sacrebleu_tensor = torch.FloatTensor([sacrebleu_score]).cuda()
    tensor_list = [torch.empty_like(sacrebleu_tensor) for _ in range(opt.world_size)]
    torch.distributed.all_gather(tensor_list, sacrebleu_tensor)
    sacrebleu_result = torch.cat(tensor_list, dim=0).contiguous().mean()

    exactmatch_tensor = torch.FloatTensor([exactmatch]).cuda()
    tensor_list = [torch.empty_like(exactmatch_tensor) for _ in range(opt.world_size)]
    torch.distributed.all_gather(tensor_list, exactmatch_tensor)
    exactmatch_result = torch.cat(tensor_list, dim=0).contiguous().mean()

    meteor_tensor = torch.FloatTensor([meteor_score]).cuda()
    tensor_list = [torch.empty_like(meteor_tensor) for _ in range(opt.world_size)]
    torch.distributed.all_gather(tensor_list, meteor_tensor)
    meteor_result = torch.cat(tensor_list, dim=0).contiguous().mean()

    rouge_tensor = torch.FloatTensor([rouge_score]).cuda()
    tensor_list = [torch.empty_like(rouge_tensor) for _ in range(opt.world_size)]
    torch.distributed.all_gather(tensor_list, rouge_tensor)
    rouge_result = torch.cat(tensor_list, dim=0).contiguous().mean()

    f1_tenor = torch.FloatTensor([f1]).cuda()
    tensor_list = [torch.empty_like(f1_tenor) for _ in range(opt.world_size)]
    torch.distributed.all_gather(tensor_list, f1_tenor)
    f1_result = torch.cat(tensor_list, dim=0).contiguous().mean()
    
    
    total_score = sacrebleu_result + meteor_result + rouge_result + f1_result
    # return exactmatch, f1, sacrebleu_score, meteor_score, rouge_score, total_score
    return exactmatch_result, ppl, f1_result, sacrebleu_result, meteor_result, rouge_result, total_score
