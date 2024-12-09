import datasets
import pandas
import sklearn
import transformers
import re



def load_data(file_path='filtered.parquet'):
    return pandas.read_parquet(file_path)


def preprocess_text(dataframe):
    def replace_vehicle_with_vn(input_string):
        return re.sub(r'Vehicle\s+#?(\d+)', r'V\1', input_string)

    dataframe['text'] = dataframe['Summary'].apply(replace_vehicle_with_vn)
    print(dataframe)
    dataframe['labels'] = dataframe['MaxVehicleSeverity'] # MaxOccupantInjurySeverity
    return dataframe


def initialize_tokenizer():
    tokenizer = transformers.AutoTokenizer.from_pretrained('openai-community/gpt2-xl')
    vehicle_tokens = [f'V{i}' for i in range(1, 15)]
    tokenizer.add_tokens(vehicle_tokens)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer


def initialize_model(tokenizer, dataframe):
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        'openai-community/gpt2-xl', num_labels=dataframe['labels'].nunique()
    )
    model.config.pad_token_id=tokenizer.convert_tokens_to_ids('[PAD]')
    model.resize_token_embeddings(len(tokenizer))
    return model


def tokenize_data(examples, tokenizer):
    return tokenizer(examples['text'], padding='max_length', truncation=True)


def prepare_dataset(dataframe, tokenizer):
    hf_dataset = datasets.Dataset.from_pandas(dataframe[['text', 'labels']])
    tokenized_dataset = hf_dataset.map(lambda x: tokenize_data(x, tokenizer), batched=True)
    return tokenized_dataset.train_test_split(test_size=0.2)


def create_training_args(learning_rate, batch_size, epochs, weight_decay):
    return transformers.TrainingArguments(
        output_dir='./results',
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        report_to='tensorboard'
    )


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    acc = sklearn.metrics.accuracy_score(labels, predictions)
    print(
        f'Accuracy: {100 * acc:.2f}%, Precision: {100 * precision:.2f}%, Recall: {100 * recall:.2f}%, F1: {100 * f1:.2f}%')
    return {'f1': f1}


def hyperparameter_search_space(trial):
    return {
        'learning_rate': trial.suggest_float('learning_rate', 5e-6, 3e-5, log=True),
        'per_device_train_batch_size': trial.suggest_categorical('per_device_train_batch_size', [4, 8, 12, 16]),
        'num_train_epochs': trial.suggest_int('num_train_epochs', 3, 10),
        'weight_decay': trial.suggest_float('weight_decay', 0.01, 0.3),
    }


def create_trainer(model_init, training_args, tokenized_dataset):
    return transformers.Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        compute_metrics=compute_metrics,
    )


def main():

    dataframe = load_data()
    dataframe = preprocess_text(dataframe)

    tokenizer = initialize_tokenizer()
    model_init_func = lambda: initialize_model(tokenizer, dataframe)

    tokenized_dataset = prepare_dataset(dataframe, tokenizer)

    trainer = create_trainer(model_init_func, None, tokenized_dataset)

    # best_run = trainer.hyperparameter_search(
    #     hp_space=hyperparameter_search_space,
    #     direction='maximize',
    #     n_trials=5,
    #     backend='optuna',
    # )

    # print("Best hyperparameters:", best_run.hyperparameters)

    training_args = create_training_args(
        1e-5,
        4,
        10,
        0.1
    )

    trainer = create_trainer(model_init_func, training_args, tokenized_dataset)

    trainer.train()
    print(trainer.evaluate())


if __name__ == '__main__':
    main()
