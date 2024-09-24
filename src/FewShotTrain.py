import tensorflow as tf
import numpy as np

def train_few_shot(model, episodes, n_classes, n_support, n_query, train_generator, val_generator, optimizer):
    for episode in range(episodes):
        
        support_set, query_set, support_labels, query_labels = generate_few_shot_samples(train_generator, n_support, n_query)

        if support_set.shape[0] == 0 or query_set.shape[0] == 0:
            print(f"Episode {episode + 1}: Not enough samples to train.")
            continue

        
        with tf.GradientTape() as tape:
            support_preds = model(support_set, training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(support_labels, support_preds)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        
        query_preds = model(query_set, training=False)
        query_loss = tf.keras.losses.sparse_categorical_crossentropy(query_labels, query_preds)

        print(f'Episode {episode + 1}/{episodes}, Loss: {loss.numpy().mean()}, Query Loss: {query_loss.numpy().mean()}')

        
        if episode % 10 == 0:  
            val_support_set, val_query_set, val_support_labels, val_query_labels = generate_few_shot_samples(val_generator, n_support, n_query)

            if val_support_set.shape[0] == 0 or val_query_set.shape[0] == 0:
                print(f"Validation at Episode {episode + 1}: Not enough samples.")
                continue

            val_preds = model(val_query_set, training=False)
            val_loss = tf.keras.losses.sparse_categorical_crossentropy(val_query_labels, val_preds)

            print(f'Validation Loss after Episode {episode + 1}: {val_loss.numpy().mean()}')
        

def generate_few_shot_samples(generator, n_support=5, n_query=5):
    
    X_support, y_support = generator.__getitem__(0)  
    X_query, y_query = generator.__getitem__(0)

    
    support_indices = np.random.choice(X_support.shape[0], n_support, replace=False)
    query_indices = np.random.choice(X_query.shape[0], n_query, replace=False)

    support_set = X_support[support_indices]
    query_set = X_query[query_indices]
    
    support_labels = np.argmax(y_support[support_indices], axis=1)
    query_labels = np.argmax(y_query[query_indices], axis=1)

    return support_set, query_set, support_labels, query_labels


def generate_test_samples(generator, n_support=5, n_query=5):
    
    X_test, y_test = generator.__getitem__(0)  # First Batch
    if X_test.shape[0] < (n_support + n_query):
        raise ValueError("Not enough samples in the test set for the required support and query sizes.")

    support_indices = np.random.choice(X_test.shape[0], n_support, replace=False)
    query_indices = np.random.choice(X_test.shape[0], n_query, replace=False)

    support_set = X_test[support_indices]
    query_set = X_test[query_indices]
    
    support_labels = np.argmax(y_test[support_indices], axis=1)
    query_labels = np.argmax(y_test[query_indices], axis=1)

    return support_set, query_set, support_labels, query_labels

def evaluate_model(model, test_generator, n_support=5, n_query=5):
    support_set, query_set, support_labels, query_labels = generate_test_samples(test_generator, n_support, n_query)

    predictions = model(query_set, training=False)

    predicted_labels = np.argmax(predictions, axis=1)

    accuracy = np.mean(predicted_labels == query_labels)

    print(f'Accuracy: {accuracy:.4f}')

