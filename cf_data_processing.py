import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Add, Activation, Lambda, BatchNormalization, Concatenate, Dropout, Input, Embedding, Dot, Reshape, Dense, Flatten
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, ReduceLROnPlateau
from itertools import product

INPUT_DIR = './Data'
rating_df = pd.read_csv(INPUT_DIR + '/reviews_data.csv', 
                        low_memory=False, 
                        usecols=["user_id", "shoes_id", "rating"]
                        )

rating_df = rating_df.sample(frac=1, random_state=1)

n_ratings = rating_df['user_id'].value_counts()
rating_df = rating_df[rating_df['user_id'].isin(n_ratings[n_ratings >= 1].index)].copy()

min_rating = min(rating_df['rating'])
max_rating = max(rating_df['rating'])
rating_df['rating'] = rating_df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values.astype(np.float64)

AvgRating = np.mean(rating_df['rating'])

duplicates = rating_df.duplicated()

if duplicates.sum() > 0:
    rating_df = rating_df[~duplicates]

# Encoding categorical data
user_ids = rating_df["user_id"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
user_encoded2user = {i: x for i, x in enumerate(user_ids)}
rating_df["user"] = rating_df["user_id"].map(user2user_encoded)
n_users = len(user2user_encoded)

shoes_ids = rating_df["shoes_id"].unique().tolist()
shoes2shoes_encoded = {x: i for i, x in enumerate(shoes_ids)}
shoes_encoded2shoes = {i: x for i, x in enumerate(shoes_ids)}
rating_df["shoes"] = rating_df["shoes_id"].map(shoes2shoes_encoded)
n_shoes = len(shoes2shoes_encoded)

X = rating_df[['user', 'shoes']].values
y = rating_df["rating"]

test_set_size = 10000 #10k for test set
train_indices = rating_df.shape[0] - test_set_size 

X_train, X_test, y_train, y_test = (
    X[:train_indices],
    X[train_indices:],
    y[:train_indices],
    y[train_indices:],
)

X_train_array = [X_train[:, 0], X_train[:, 1]]
X_test_array = [X_test[:, 0], X_test[:, 1]]

def RecommenderNet():
   embedding_size = 128
   
   user = Input(name = 'user', shape = [1])
   user_embedding = Embedding(name = 'user_embedding',
                     input_dim = n_users, 
                     output_dim = embedding_size)(user)
   
   shoes = Input(name = 'shoes', shape = [1])
   shoes_embedding = Embedding(name = 'shoes_embedding',
                     input_dim = n_shoes, 
                     output_dim = embedding_size)(shoes)
   
   x = Dot(name = 'dot_product', normalize = True, axes = 2)([user_embedding, shoes_embedding])
   x = Flatten()(x)
      
   x = Dense(1, kernel_initializer='he_normal')(x)
   x = BatchNormalization()(x)
   x = Activation("sigmoid")(x)
   
   model = Model(inputs=[user, shoes], outputs=x)
   model.compile(loss='binary_crossentropy', metrics=["mae", "mse"], optimizer='Adam')
   
   return model

try:
   tpu = tf.distribute.cluster_resolver.TPUClusterResolver() 
   tf.config.experimental_connect_to_cluster(tpu)
   tf.tpu.experimental.initialize_tpu_system(tpu)
   strategy = tf.distribute.experimental.TPUStrategy(tpu)
except ValueError:
   strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
   model = RecommenderNet()

start_lr = 0.00001
min_lr = 0.00001
max_lr = 0.00005
batch_size = 10000

# if TPU_INIT:
max_lr = max_lr * strategy.num_replicas_in_sync
batch_size = batch_size * strategy.num_replicas_in_sync

rampup_epochs = 5
sustain_epochs = 0
exp_decay = .8

def lrfn(epoch):
    if epoch < rampup_epochs:
        return (max_lr - start_lr)/rampup_epochs * epoch + start_lr
    elif epoch < rampup_epochs + sustain_epochs:
        return max_lr
    else:
        return (max_lr - min_lr) * exp_decay**(epoch-rampup_epochs-sustain_epochs) + min_lr


lr_callback = LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=0)

checkpoint_filepath = 'models/shoes_weights_checkpoint.weights.h5'

model_checkpoints = ModelCheckpoint(filepath=checkpoint_filepath,
                                        save_weights_only=True,
                                        monitor='val_loss',
                                        mode='min',
                                        save_best_only=True)

early_stopping = EarlyStopping(patience = 3, monitor='val_loss', mode='min', restore_best_weights=True)

my_callbacks = [
    model_checkpoints,
    lr_callback,
    early_stopping,   
]

from tensorflow.keras.models import load_model
import os 

model_path = 'models/shoes_model.h5'
history = {} # Initialize history

if not os.path.exists(model_path):
   history = model.fit(
       x=X_train_array,
       y=y_train,
       batch_size=batch_size,
       epochs=20,
       verbose=1,
       validation_data=(X_test_array, y_test),
       callbacks=my_callbacks
   )
   model.save(model_path)
   model.load_weights(checkpoint_filepath)
else:
   model = load_model(model_path)
   model.load_weights(checkpoint_filepath)

def extract_weights(name, model):
    weight_layer = model.get_layer(name)
    weights = weight_layer.get_weights()[0]
    weights = weights / np.linalg.norm(weights, axis = 1).reshape((-1, 1))
    return weights

shoes_weights = extract_weights('shoes_embedding', model)
user_weights = extract_weights('user_embedding', model)

pd.set_option("max_colwidth", None)

def find_similar_users(item_input, n=10,return_dist=False, neg=False):
    try:
        index = item_input
        encoded_index = user2user_encoded.get(index)
        weights = user_weights
    
        dists = np.dot(weights, weights[encoded_index])
        sorted_dists = np.argsort(dists)
        
        n = n + 1
        
        if neg:
            closest = sorted_dists[:n]
        else:
            closest = sorted_dists[-n:]

        if return_dist:
            return dists, closest
        
        SimilarityArr = []
        
        for close in closest:
            similarity = dists[close]

            if isinstance(item_input, int):
                decoded_id = user_encoded2user.get(close)
                if(decoded_id != index):
                    SimilarityArr.append({"similar_users": decoded_id, 
                                        "similarity": similarity})

        Frame = pd.DataFrame(SimilarityArr).sort_values(by="similarity", 
                                                        ascending=False)
        return Frame
    
    except:
        print('{}!, Not Found in User list')

def random_user_picker():
  ratings_per_user = rating_df.groupby('user_id').size()
  random_user = ratings_per_user[ratings_per_user < 500].sample(1, random_state=None).index[0]
  return random_user

# similar_users = find_similar_users(int(random_user_picker()), 
#                                    n=5, 
#                                    neg=False)

# similar_users = similar_users[similar_users.similarity > 0.2]
# print(similar_users)

def pred_ranking_based_recommendation(user, page=1, page_size=10):

    start_index = (page - 1) * page_size
    end_index = start_index + page_size

    all_combinations = list(product([user2user_encoded.get(user)], shoes2shoes_encoded.keys()))

    encoded_combinations = [(user_encoder, shoes2shoes_encoded[shoes]) for user_encoder, shoes in all_combinations]

    user_inputs = np.array([encoded_combinations[i][0] for i in range(len(encoded_combinations))])
    shoes_inputs = np.array([encoded_combinations[i][1] for i in range(len(encoded_combinations))])

    ratings = model.predict([user_inputs, shoes_inputs]).flatten()

    top_ratings_indices = (-ratings).argsort()[start_index:end_index]

    recommended_shoes_id = [list(shoes2shoes_encoded.keys())[index] for index in top_ratings_indices]

    recommended_scores = ratings[top_ratings_indices]
  
    total_shoes = len(all_combinations)

    total_pages = total_shoes // page_size
    if total_shoes % page_size > 0:
        total_pages += 1

    return list(zip(recommended_shoes_id, recommended_scores)), total_pages


# pred_ranking_based_recommendations = pred_ranking_based_recommendation(random_user_picker())
# print(pred_ranking_based_recommendations)