from lightning_sdk import Studio, Job, Machine

studio = Studio()

job1 = Job.run(
    name="train-qdora-seed43-diagnostic-bf16",
    command=(
        "cd /teamspace/studios/this_studio/dora-case-studies && "
        "python train.py --config ../train_configs/qdora-diagnostic-bf16-seed43.yaml"
    ),
    studio=studio,
    machine=Machine.H100,
)


job2 = Job.run(
    name="train-qdora-seed43-diagnostic-fp32",
    command=(
        "cd /teamspace/studios/this_studio/dora-case-studies && "
        "python train.py --config ../train_configs/qdora-diagnostic-fp32-seed43.yaml"
    ),
    studio=studio,
    machine=Machine.H100,
)