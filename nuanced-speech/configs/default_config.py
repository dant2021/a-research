config = {
    'batch_size': 8,
    'gradient_accumulation_steps': 4,
    'learning_rate': 3e-4,
    'warmup_steps': 1000,
    'max_steps': 100000,
    'eval_steps': 1000,
    'save_steps': 5000,
    'whisper_model': 'base',
    'freeze_whisper': True,
    'num_gpu': 2,
    'precision': 'fp16',
    'whisper_hidden_dim': 768,  # for base model
    'kokoro_voice': 'af_heart',  # Kokoro voice to use
    'kokoro_lang_code': 'a',     # 'a' for American English
    'feature_loss_weight': 0.1,
} 