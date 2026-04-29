import os
import kinetic

# --- Configuration ---
vit_config = {
    "num_classes": 10,
    "input_shape": (64, 64, 3),
    "patch_size": 4,
    "num_patches": (64 // 4) ** 2,
    "projection_dim": 64,
    "num_heads": 4,
    "transformer_units": [128, 64],
    "transformer_layers": 8,
    "mlp_head_units": [2048, 1024],
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 256,
    "num_epochs": 100,
}

@kinetic.run(
    accelerator="l4",
    container_image="bundled",
)
def train_vit_eurosat():
    # --- Backend Configuration ---
    import os
    os.environ["KERAS_BACKEND"] = "tensorflow"
    
    # Ensure NVIDIA libraries are visible in GKE environment
    os.environ["LD_LIBRARY_PATH"] = "/usr/local/nvidia/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["NVIDIA_VISIBLE_DEVICES"] = "all"
    os.environ["NVIDIA_DRIVER_CAPABILITIES"] = "all"

    # --- Imports ---
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import kagglehub

    print(f"\n[Kinetic] Starting job on: {os.uname().nodename}")
    print(f"[Kinetic] Keras Backend: {os.environ.get('KERAS_BACKEND')}")
    print(f"[Kinetic] GPUs available: {tf.config.list_physical_devices('GPU')}")

    output_dir = os.environ.get("KINETIC_OUTPUT_DIR", "./output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[Kinetic] Artifacts will be saved to: {output_dir}")

    # --- Data Preparation ---
    print("\n--- Phase 1: Data Preparation ---")
    print("Downloading EuroSAT dataset...")
    path = kagglehub.dataset_download("apollo2506/eurosat-dataset")
    data_path = os.path.join(path, "EuroSAT")
    print(f"Dataset path: {data_path}")

    # Load dataset
    print("Loading images from directory...")
    ds = tf.keras.utils.image_dataset_from_directory(
        data_path,
        label_mode="categorical",
        image_size=(64, 64),
        batch_size=vit_config["batch_size"],
        shuffle=True,
        seed=42,
        validation_split=0.2,
        subset="both"
    )
    train_ds, val_ds = ds
    print(f"Loaded {len(train_ds)} training batches")
    print(f"Loaded {len(val_ds)} validation batches")

    # Data Augmentation
    print("Configuring data augmentation pipeline...")
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.1),
    ])

    # --- Model Components ---
    def mlp(x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x

    class ViTPatchEmbedding(layers.Layer):
        def __init__(self, patch_size, num_patches, embed_dim, **kwargs):
            super().__init__(**kwargs)
            self.projection = layers.Conv2D(
                filters=embed_dim,
                kernel_size=patch_size,
                strides=patch_size,
                padding="valid"
            )
            self.reshape = layers.Reshape(target_shape=(num_patches, embed_dim))
            self.cls_token = self.add_weight(
                name="cls_token",
                shape=(1, 1, embed_dim),
                initializer="zeros",
                trainable=True
            )
            self.pos_embedding = layers.Embedding(
                input_dim=num_patches + 1, output_dim=embed_dim
            )
            self.num_patches = num_patches

        def call(self, x):
            batch_size = tf.shape(x)[0]
            # Projection and Reshape
            x = self.projection(x)
            x = self.reshape(x)
            
            # Tile CLS token for the batch
            cls_tokens = tf.tile(self.cls_token, [batch_size, 1, 1])
            x = tf.concat([cls_tokens, x], axis=1)
            
            # Positional Encoding
            positions = tf.range(start=0, limit=self.num_patches + 1, delta=1)
            x = x + self.pos_embedding(positions)
            return x

    def create_vit_classifier():
        inputs = layers.Input(shape=vit_config["input_shape"])
        augmented = data_augmentation(inputs)
        
        # Patching and Embedding
        embeddings = ViTPatchEmbedding(
            vit_config["patch_size"], 
            vit_config["num_patches"], 
            vit_config["projection_dim"]
        )(augmented)

        # Transformer Blocks
        for _ in range(vit_config["transformer_layers"]):
            # Layer normalization 1.
            x1 = layers.LayerNormalization(epsilon=1e-6)(embeddings)
            # Create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(
                num_heads=vit_config["num_heads"], 
                key_dim=vit_config["projection_dim"], 
                dropout=0.1
            )(x1, x1)
            # Skip connection 1.
            x2 = layers.Add()([attention_output, embeddings])
            # Layer normalization 2.
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP.
            x3 = mlp(x3, hidden_units=vit_config["transformer_units"], dropout_rate=0.1)
            # Skip connection 2.
            embeddings = layers.Add()([x3, x2])

        # Classification Head
        representation = layers.LayerNormalization(epsilon=1e-6)(embeddings)
        representation = layers.Lambda(lambda x: x[:, 0])(representation) # Take [CLS] token
        representation = layers.Dropout(0.5)(representation)
        
        features = mlp(representation, hidden_units=vit_config["mlp_head_units"], dropout_rate=0.5)
        logits = layers.Dense(vit_config["num_classes"])(features)
        
        return keras.Model(inputs=inputs, outputs=logits)

    # --- Training ---
    print("\n--- Phase 2: Model Compilation & Training ---")
    model = create_vit_classifier()
    print("Vision Transformer model built successfully.")
    
    optimizer = keras.optimizers.AdamW(
        learning_rate=vit_config["learning_rate"], 
        weight_decay=vit_config["weight_decay"]
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    checkpoint_filepath = os.path.join(output_dir, "vit_checkpoint.keras")
    print(f"Model checkpoints will be saved to: {checkpoint_filepath}")

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_best_only=True,
            monitor="val_accuracy",
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", 
            patience=10, 
            restore_best_weights=True
        )
    ]

    print(f"Starting training for {vit_config['num_epochs']} epochs...")

    history = model.fit(
        train_ds,
        batch_size=vit_config["batch_size"],
        epochs=vit_config["num_epochs"],
        validation_data=val_ds,
        callbacks=callbacks,
    )

    # Save final model
    final_path = os.path.join(output_dir, "vit_model_final.keras")
    model.save(final_path)
    print("\n--- Training Complete ---")
    print(f"Final model saved to: {final_path}")
    return history.history

if __name__ == "__main__":
    train_vit_eurosat()