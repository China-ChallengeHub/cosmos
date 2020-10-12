import time
import visdom
import numpy as np
# env = 'my_windows'
env = "commonsenseqa"
viz = visdom.Visdom(env=env)

win_epoch_loss = viz.line(
    X=np.array([0]),
    # Y=np.array([0]),
    Y=np.column_stack((np.array(0), np.array(0))),
    opts=dict(
        title="epoch_loss",
        legend=["train", "eval"],
        showlegend=True,
        xlabel="epoch",
        ylabel="loss"
    )
)

win_eval_accuracy = viz.line(
    X=np.array([0]),
    Y=np.array([0]),
    opts=dict(
        title="accuracy",
        legend=["accuracy"],
        showlegend=True,
        xlabel="epoch",
        ylabel="rate"
    )
)


win_step_result = viz.line(
    X=np.array([0]),
    Y=np.column_stack( (np.array(0), np.array(0)) ),
    opts=dict(
        title="step_result",
        legend=["eval_loss", "eval_accuracy"],
        showlegend=True,
        xlabel="step",
        ylabel="rate"
    )
)


win_lr = viz.line(
    X=np.array([0]),
    Y=np.array([0]),
    opts=dict(
        title="lr",
        legend=["lr"],
        showlegend=True,
        xlabel="step",
        ylabel="lr"
    )
)


def update_step_result(eval_loss, eval_accuracy, global_step):
    viz.line(
        Y=np.column_stack((np.array(eval_loss), np.array(eval_accuracy))),
        X=np.array([global_step]),
        win=win_step_result,
        env=env,
        update="append"
    )
    time.sleep(0.1)


def update_epoch_loss(train_loss, eval_loss, epoch):
    viz.line(
        Y=np.column_stack((np.array(train_loss), np.array(eval_loss))),
        X=np.array([epoch]),
        win=win_epoch_loss,
        env=env,
        update="append"
    )
    time.sleep(0.1)


def update_eval_accuracy(accuracy, epoch):
    viz.line(
        Y=np.array([accuracy]),
        X=np.array([epoch]),
        win=win_eval_accuracy,
        env=env,
        update="append"
    )
    time.sleep(0.1)


def update_lr(lr_this_step, global_step):
    viz.line(
        Y=np.array([lr_this_step]),
        X=np.array([global_step]),
        win=win_lr,
        env=env,
        update="append",
        name="lr"
    )
    time.sleep(0.1)
