set -x

# OPSD-style Self-Distillation: Same model as both student and teacher
# Teacher is conditioned on ground truth answer to provide better signal
# Key: algorithm.condition_teacher_on_answer=True

python -m examples.solver_judge_distill.train_simple_math_distill_tinker \
    model.name=Qwen/Qwen3-4B-Instruct-2507 \
    model.lora_rank=32 \
    training.group_size=1 \
    training.val_group_size=4 \
    training.learning_rate=4e-5 \
    sampling.temperature=1.0 \
    sampling.top_p=1.0 \
    algorithm.adv_estimator=distill \
    algorithm.shared_tokenizer=True \
    +algorithm.condition_teacher_on_answer=True \
    +algorithm.clip_advantages=True \
    +algorithm.adv_clip_min=-5.0 \
    +algorithm.adv_clip_max=5.0 \
    algorithm.teacher_rollout_args.backend=tinker \
    algorithm.teacher_rollout_args.model=Qwen/Qwen3-4B-Instruct-2507 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.train_batch_size=64 \
    data.val_batch_size=32 \
    trainer.total_epochs=100 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='solver-judge-distill' \
    trainer.experiment_name='opsd-self-distill-4b-instruct' \
    trainer.val_before_train=True \
    trainer.test_freq=10 \
    trainer.save_freq=20 \
    trainer.default_local_dir='./outputs/opsd-self-distill-4b-instruct' \
    rollout_engine.bypass_render_with_parser=True \
    rollout_engine.disable_thinking=False \
    workflow.n_parallel_tasks=256 \
    training.rollout_timeout=1200

