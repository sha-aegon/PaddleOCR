import os
import paddle
from paddleocr.ppocr.modeling.architectures import build_model
from paddleocr.ppocr.optimizer import build_optimizer
from paddleocr.ppocr.losses import build_loss
from paddleocr.ppocr.data import build_dataloader
from paddleocr.utils.save_load import load_config, load_pretrained_params
from paddleocr.ppocr.utils.logging import logger

def main(config_path):
    # Step 1: Load Configuration
    config = load_config(config_path)

    # Update configuration for your custom dataset
    config['Global']['trainset_dir'] = "/opt/ml/input/data/training/images"
    config['Global']['label_file_list'] = ["/opt/ml/input/data/training/label.txt"]
    config['Global']['output_dir'] = "/opt/ml/model/"
    config['Global']['epoch_num'] = 100
    config['Train']['loader']['batch_size_per_card'] = 16

    # Step 2: Build Data Loaders
    train_dataloader = build_dataloader(config, 'Train', device=paddle.get_device())
    val_dataloader = build_dataloader(config, 'Eval', device=paddle.get_device())

    # Step 3: Build Model, Optimizer, and Loss
    model = build_model(config['Architecture'])
    optimizer = build_optimizer(config, model)
    loss_fn = build_loss(config)

    # Step 4: Load Pretrained Weights (Optional)
    if config['Global']['pretrained_model']:
        load_pretrained_params(model, config['Global']['pretrained_model'])

    # Step 5: Training Loop
    logger.info("Starting training...")
    for epoch in range(config['Global']['epoch_num']):
        model.train()
        for batch_idx, batch_data in enumerate(train_dataloader):
            images, labels = batch_data[0], batch_data[1]
            preds = model(images)
            loss = loss_fn(preds, labels)

            loss.backward()
            optimizer.step()
            optimizer.clear_gradients()

            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.numpy()}")

        # Validation Loop
        model.eval()
        val_loss = 0
        for batch_data in val_dataloader:
            images, labels = batch_data[0], batch_data[1]
            preds = model(images)
            val_loss += loss_fn(preds, labels).numpy()
        logger.info(f"Epoch {epoch}, Validation Loss: {val_loss}")

    # Step 6: Save Trained Model
    os.makedirs(config['Global']['output_dir'], exist_ok=True)
    paddle.save(model.state_dict(), os.path.join(config['Global']['output_dir'], "final_model.pdparams"))
    logger.info("Training complete. Model saved.")

if __name__ == "__main__":
    # Path to the PaddleOCR YAML configuration file
    config_file = "configs/rec/rec_r34_vd_none_bilstm_ctc.yml"
    main(config_file)
