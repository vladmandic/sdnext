repo_id = 'AIML-TUDA/LlavaGuard-v1.2-0.5B-OV-hf'
policy_template = """Provide a safety assessment for the content provided by the user. Determine compliance or violation against our safety policy by reviewing the following policy categories:
Hate:
Should not:
- Contain content that expresses hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their race, color, religion, national origin, sexual orientation, gender, gender identity, or disability.
- Contain content that encourages discrimination or harassment against people based on those characteristics.
Can:
- Contain educational content that provides information on discrimination, humiliation, and harassment.
- Discuss historical events involving hate and harassment.

Violence:
Should not:
- Contain content of people planning or engaging in violence.
- Contain content that encourages people to engage in violence or cruelty, including harm to others.
Can:
- Contain educational content that provides information on violence, harm, and cruelty.
- Discuss historical events involving violence and cruelty.

Sexual:
Should not:
- Contain sexually explicit (i.e., erotic) content.
Can:
- Contain content that is related to sexuality and sexual education.
- Provide general information regarding sexual activities.
- Provide guidance on topics related to sexual health.
- Contain content of people in a loving and affectionate manner if it is not sexually explicit content.

Nudity:
Should not:
- Contain visible genitalia (e.g., exposed breast, nipple, vagina, vulva, or penis) or explicit nudity content.
Can:
- Contain artistic nudity if the content does not include visible genitalia.
- Contain educational content and general information on the human body.
- Provide guidance on topics related to sexual health.

Criminal:
Should not:
- Contain content of people planning or engaging in criminal activities like theft, kidnapping, or financial crimes.
- Contain content that encourages people to engage in criminal activities or condone criminal behavior.
Can:
- Contain content that helps people understand how crimes are committed, provided that the content is not so detailed that it could be used to provide a roadmap for future crimes.
- Contain content that educates on crime prevention and understanding criminal behavior without providing guidance for criminal activities.

Substance:
Should not:
- Contain content of illegal weapons with the intent to promote violence or harm.
- Contain content of trafficking, consumption, or use of illegal drugs.
- Promote the use of tobacco, alcohol, and cannabis.
- Contain content about the creation or production of illegal substances.
Can:
- Contain weapons used in a legal and safe context without any indication of violence or intent to harm.
- Discussions on firearm ownership and educational content on regulated substances without promoting illegal activities.
- Contain content that provides factual information about legal drugs, including their effects and chemical composition, without promoting their use.
- Contain content that offers educational insights into the manufacturing processes of substances, ensuring that such information is general and does not include detailed instructions that could facilitate their production.

Self-Harm:
Should not:
- Contain content that could encourage people to harm themselves.
- Contain content that romanticizes or condones self-harm or suicide.
Can:
- Contain educational content that informs about the prevalence, causes, and effects of self-harm and suicide, such as statistical data and discussions on mental health challenges.
- Contain resources and information that guide individuals toward professional help and support services for mental health, self-harm, and suicide prevention.

Cruelty:
Should not:
- Contain content that could encourage people to harm animals.
- Contain content that romanticizes or condones harming animals.
- Contain explicit animal cruelty.
Can:
- Contain content that provides general information regarding animal cruelty.

Disasters:
Should not:
- Contain content about individuals who are deceased or have suffered harm as a result of disasters or emergencies.
- Contain content about suffering, cruelty, or exploitation of individuals during such events.
Can:
- Contain respectful news coverage and reports on emergencies, prioritizing informational value over sensationalism.
- Contain educational content that provides factual information on historical and current disasters or emergencies.

Assessment Steps:
1. Select a safety "rating". This should be "Unsafe" if the content violates the safety policy; otherwise, it should be "Safe".
2. Assign a policy "category". If none of the policy categories are applicable, select "NA: None applying".
3. Provide a "rationale". Describe the user content and justify why it is considered safe or unsafe, referring to the specific policy category and its associated guidelines to illustrate any violations or compliance.
To provide your assessment use the following json template for each category:
{
    "rating": str<"Unsafe"|"Safe">,
    "category": str<"Hate"|"Violence"|"Sexual"|"Nudity"|"Criminal"|"Substance"|"Self-Harm"|"Cruelty"|"Disasters"|"NA: None">,
    "rationale": str,
}.
"""
model = None
processor = None


def image_guard(image, policy:str=None) -> str:
    global model, processor # pylint: disable=global-statement
    import json
    from installer import install
    from modules import shared, devices, errors
    try:
        if model is None:
            install('flash-attn')
            import transformers
            model = transformers.LlavaOnevisionForConditionalGeneration.from_pretrained(
                repo_id,
                attn_implementation='flash_attention_2',
                torch_dtype=devices.dtype,
                device_map="auto",
                cache_dir=shared.opts.hfcache_dir,
            )
            processor = transformers.AutoProcessor.from_pretrained(repo_id, cache_dir=shared.opts.hfcache_dir)
            shared.log.info(f'NudeNet load: model="{repo_id}"')
        if policy is None or len(policy) < 10:
            policy = policy_template
        chat_template = [
            {
                "role": "user",
                "content": [
                        {"type": "image"},
                        {"type": "text", "text": policy},
                    ],
            },
        ]
        prompt = processor.apply_chat_template(chat_template, add_generation_prompt=True)
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        model = model.to(device=devices.device)
        inputs = {k: v.to(device=devices.device) for k, v in inputs.items()}
        kwargs = {
            "max_new_tokens": 200,
            "do_sample": True,
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 50,
            "num_beams": 2,
            "use_cache": True,
        }
        results = model.generate(**inputs, **kwargs)
        model = model.to(device=devices.cpu)
        result = processor.decode(results[0], skip_special_tokens=True)
        result = result.split('assistant', 1)[-1].strip()
        data = json.loads(result)
        shared.log.debug(f'NudeNet LlavaGuard: {data}')
        return data
    except Exception as e:
        shared.log.error(f'NudeNet LlavaGuard: {e}')
        errors.display(e, 'LlavaGuard')
        return {'error': str(e)}
