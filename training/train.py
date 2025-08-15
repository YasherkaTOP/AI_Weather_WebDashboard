from utils import get_dataloaders_and_model, get_lr
from config import GEO_POINTS, MAX_EPOCHS
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
import os
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    pl.seed_everything(42)
    train, test, train_dataset = get_dataloaders_and_model(GEO_POINTS)

    # initial hparameters
    model = TemporalFusionTransformer.from_dataset(
        train_dataset,
        learning_rate=0.03,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.15,
        hidden_continuous_size=32,
        loss=QuantileLoss(),
        reduce_on_plateau_patience=2,
    )

    lr = get_lr(model, train, test)

    if not lr:
        lr = 0.01

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=3, verbose=False, mode="min")
    checkpoint_callback = ModelCheckpoint(save_top_k=3, monitor="val_loss")
    trainer = pl.Trainer(
        accelerator="auto",
        gradient_clip_val=0.1,
        max_epochs=MAX_EPOCHS,
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    model = TemporalFusionTransformer.from_dataset(
        train_dataset,
        learning_rate=0.01,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.15,
        hidden_continuous_size=32,
        loss=QuantileLoss(),
        reduce_on_plateau_patience=2,
    )

    trainer.fit(
        model, train_dataloaders=train, val_dataloaders=test,
    )

    os.rename(checkpoint_callback.best_model_path, 'result/model.ckpt')
