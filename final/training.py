import re

import datasets
import matplotlib.pyplot as plt
import numpy
import pandas
import seaborn
import sklearn
import transformers

MODEL_NAME = 'bert-base-uncased'

DATASET_PATH = 'filtered.parquet'
RESULT_OUTPUT_PATH = './results'

LABEL_TO_PREDICT = 'MaxVehicleSeverity'  # or MaxOccupantInjurySeverity
NUM_LABELS = 4 if LABEL_TO_PREDICT == 'MaxVehicleSeverity' else 7
AUGMENT_SUMMARIES = True

LR_MIN = 5e-6
LR_MAX = 5e-5
BATCH_SIZES = [8, 10, 12]
EPOCHS_MIN = 3
EPOCHS_MAX = 10
WEIGHT_DECAY_MIN = 0.10
WEIGHT_DECAY_MAX = 0.30

HP_OPTIMIZATION_TRIALS = 10


def main():
    dataset = pandas.read_parquet(DATASET_PATH)
    dataset = preprocess(dataset)

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model_init = lambda: create_model(tokenizer)

    dataset = tokenize_dataset(dataset, tokenizer)

    trainer = create_trainer(model_init, None, dataset)

    train(trainer, model_init, dataset)


def preprocess(dataset):
    def replace_vehicle_names(summary):
        return re.sub(r'V#?(\d+)', r'Vehicle #\1', summary)

    dataset['Summary'] = dataset['Summary'].apply(replace_vehicle_names)

    def augment_summaries(row):
        augment = ''

        match LABEL_TO_PREDICT:
            case 'MaxVehicleSeverity':
                delta_v = row['DeltaV']
                delta_v = 'unknown' if numpy.isnan(delta_v) else f'{delta_v} km/h'
                augment = f'The estimated delta V on collision is {delta_v}.'

            case 'MaxOccupantInjurySeverity':
                max_injuries = row['MaxInjuryCount']
                max_injuries = 'unknown' if numpy.isnan(max_injuries) else int(max_injuries)
                augment = f'The maximum number of injuries sustained in this crash is {max_injuries}.'

        return augment + ' ' + row['Summary']

    if AUGMENT_SUMMARIES:
        dataset['Summary'] = dataset.apply(augment_summaries, axis=1)

    dataset['text'] = dataset['Summary']
    dataset['labels'] = dataset[LABEL_TO_PREDICT]

    dataset = dataset.dropna(subset=['text', 'labels'])
    dataset['labels'] = dataset['labels'].astype(numpy.uint8)

    return dataset


def create_model(tokenizer):
    model = transformers.AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model.resize_token_embeddings(len(tokenizer))
    return model


def tokenize_dataset(dataset, tokenizer):
    dataset = datasets.Dataset.from_pandas(dataset[['text', 'labels']])

    def tokenize(samples):
        return tokenizer(samples['text'], padding='max_length', truncation=True)

    dataset = dataset.map(tokenize, batched=True)

    return dataset.train_test_split(test_size=0.2)


def create_trainer(model_init, training_args, dataset):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)

        precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(labels, predictions,
                                                                                   average='weighted', zero_division=0)
        accuracy = sklearn.metrics.accuracy_score(labels, predictions)

        confusion_matrix = sklearn.metrics.confusion_matrix(labels, predictions)

        plt.figure(figsize=(10, 8))
        seaborn.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(f'Confusion Matrix (F1: {100 * f1:.2f}%)')
        plt.show()

        print(f'Accuracy:  {100 * accuracy:.2f}%,'
              f'Precision: {100 * precision:.2f}%,'
              f'Recall:    {100 * recall:.2f}%,'
              f'F1:        {100 * f1:.2f}%')

        return {'f1': f1}

    return transformers.Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        compute_metrics=compute_metrics,
    )


def train(trainer, model_init, dataset):
    def hp_search_space(trial):
        return {
            'learning_rate': trial.suggest_float('learning_rate', LR_MIN, LR_MAX, log=True),
            'per_device_train_batch_size': trial.suggest_categorical('per_device_train_batch_size', BATCH_SIZES),
            'num_train_epochs': trial.suggest_int('num_train_epochs', EPOCHS_MIN, EPOCHS_MAX),
            'weight_decay': trial.suggest_float('weight_decay', WEIGHT_DECAY_MIN, WEIGHT_DECAY_MAX),
        }

    best_run = trainer.hyperparameter_search(
        hp_space=hp_search_space,
        direction='maximize',
        n_trials=HP_OPTIMIZATION_TRIALS,
        backend='optuna',
    )

    print(f'Best hyperparameters: {best_run.hyperparameters}')

    training_args = transformers.TrainingArguments(
        output_dir=RESULT_OUTPUT_PATH,
        learning_rate=best_run.hyperparameters['learning_rate'],
        per_device_train_batch_size=best_run.hyperparameters['per_device_train_batch_size'],
        num_train_epochs=best_run.hyperparameters['num_train_epochs'],
        weight_decay=best_run.hyperparameters['weight_decay'],
        save_strategy='epoch',
        eval_strategy='epoch',
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        fp16=True,
    )

    trainer = create_trainer(model_init, training_args, dataset)

    trainer.train()

    print(trainer.evaluate())


if __name__ == '__main__':
    main()
