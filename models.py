from keras.layers.embeddings import Embedding
from keras.models import Sequential, clone_model, model_from_json
from keras.optimizers import Adam
from keras import Input, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, SpatialDropout1D, Bidirectional, Dense, \
    LSTM, Conv1D, MaxPooling1D, Dropout, concatenate, Flatten, add
from keras.layers import LSTM as CuDNNLSTM
from utils import Attention


def build_model1(X_train, y_train, X_valid, y_valid, max_len, max_features, embed_size, embedding_matrix, lr=0.0, lr_d=0.0, spatial_dr=0.0, dense_units=128, conv_size=128, dr=0.2, patience=3, fold_id=1):
    file_path = f"best_model_fold_{fold_id}.hdf5"
    check_point = ModelCheckpoint(
        file_path, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")
    early_stop = EarlyStopping(
        monitor="val_accuracy", mode="max", patience=patience)
    inp = Input(shape=(max_len,))
    x = Embedding(max_features + 1, embed_size * 2,
                  weights=[embedding_matrix], trainable=False)(inp)
    x1 = SpatialDropout1D(spatial_dr)(x)
    att = Attention(max_len)(x1)
    x = Conv1D(conv_size, 2, activation='relu', padding='same')(x1)
    x = MaxPooling1D(5, padding='same')(x)
    x = Conv1D(conv_size, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(5, padding='same')(x)
    x = Flatten()(x)
    x = concatenate([x, att])
    x = Dropout(0.5)(Dense(dense_units, activation='relu')(x))
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss="binary_crossentropy",
                  optimizer=Adam(), metrics=["accuracy"])
    model2 = Model(inputs=inp, outputs=x)
    model.fit(X_train, y_train, batch_size=16, epochs=20, validation_data=(
        X_valid, y_valid), verbose=1, callbacks=[early_stop, check_point])
    model2.load_weights(file_path)
    model2.compile(loss="binary_crossentropy",
                   optimizer=Adam(), metrics=["accuracy"])
    return model2


def build_model3(X_train, y_train, X_valid, y_valid, max_len, max_features, embed_size, embedding_matrix, lr=0.0, lr_d=0.0, spatial_dr=0.0, dense_units=128, conv_size=128, dr=0.2, patience=3, fold_id=1):
    file_path = f"best_model_fold_{fold_id}.hdf5"
    check_point = ModelCheckpoint(
        file_path, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")
    early_stop = EarlyStopping(
        monitor="val_accuracy", mode="max", patience=patience)
    main_input = Input(shape=(max_len,), name='main_input')
    glove_Embed = (Embedding(max_features + 1, embed_size * 2,
                             weights=[embedding_matrix], trainable=False))(main_input)
    y = LSTM(300)(glove_Embed)
    y = Dropout(rate=0.5)(y)
    y = Dense(200, activation='relu')(y)
    y = Dropout(rate=0.5)(y)
    z = Dense(100, activation='relu')(y)
    output_lay = Dense(1, activation='sigmoid')(z)
    model = Model(inputs=[main_input], outputs=[output_lay])
    model.compile(loss="binary_crossentropy",
                  optimizer=Adam(), metrics=["accuracy"])
    model2 = Model(inputs=[main_input], outputs=[output_lay])
    model.fit(X_train, y_train, batch_size=16, epochs=20, validation_data=(
        X_valid, y_valid), verbose=1, callbacks=[early_stop, check_point])
    model2.load_weights(file_path)
    model2.compile(loss="binary_crossentropy",
                   optimizer=Adam(), metrics=["accuracy"])
    return model2


def build_model4(X_train, y_train, X_valid, y_valid, max_len, max_features, embed_size, embedding_matrix, lr=0.0, lr_d=0.0, spatial_dr=0.0, dense_units=128, conv_size=128, dr=0.2, patience=3, fold_id=1):
    file_path = f"best_model_fold_{fold_id}.hdf5"
    check_point = ModelCheckpoint(
        file_path, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")
    early_stop = EarlyStopping(
        monitor="val_accuracy", mode="max", patience=patience)
    main_input = Input(shape=(max_len,), dtype='int32', name='main_input')
    glove_Embed = (Embedding(max_features + 1, embed_size * 2, input_length=max_len,
                             weights=[embedding_matrix], trainable=False))(main_input)

    x0 = Conv1D(128, 10, activation='relu')(glove_Embed)
    x1 = Conv1D(64, 5, activation='relu')(x0)
    x2 = Conv1D(32, 4, activation='relu')(x1)
    x3 = Conv1D(16, 3, activation='relu')(x2)
    x4 = Conv1D(8, 5, activation='relu')(x3)
    x = MaxPooling1D(pool_size=3)(x4)
    x = Dropout(rate=0.5)(x)
    x = LSTM(100)(x)

    p = MaxPooling1D(pool_size=10)(x0)
    p = Dropout(rate=0.5)(p)
    p = LSTM(100)(p)

    o = MaxPooling1D(pool_size=8)(x1)
    o = Dropout(rate=0.5)(o)
    o = LSTM(100)(o)

    i = MaxPooling1D(pool_size=6)(x2)
    i = Dropout(rate=0.5)(i)
    i = LSTM(100)(i)

    r = MaxPooling1D(pool_size=4)(x3)
    r = Dropout(rate=0.5)(r)
    r = LSTM(100)(r)

    t = MaxPooling1D(pool_size=3)(x4)
    t = Dropout(rate=0.5)(t)
    t = LSTM(100)(t)

    y = LSTM(500)(glove_Embed)
    y = Dense(250, activation='relu')(y)
    y = Dropout(rate=0.5)(y)

    z = concatenate([x, p, o, i, r, t, y])

    z = Dense(400, activation='relu')(z)
    z = Dropout(0.5)(z)
    z = Dense(200, activation='relu')(z)
    z = Dense(100, activation='relu')(z)
    z = Dropout(0.5)(z)
    z = Dense(50, activation='relu')(z)
    output_lay = Dense(1, activation='sigmoid')(z)
    model = Model(inputs=[main_input], outputs=[output_lay])
    model.compile(loss="binary_crossentropy", optimizer=Adam(
        lr=lr, decay=lr_d), metrics=["accuracy"])
    model2 = Model(inputs=[main_input], outputs=[output_lay])
    model.fit(X_train, y_train, batch_size=16, epochs=20, validation_data=(
        X_valid, y_valid), verbose=1, callbacks=[early_stop, check_point])
    model2.load_weights(file_path)
    model2.compile(loss="binary_crossentropy", optimizer=Adam(
        lr=lr, decay=lr_d), metrics=["accuracy"])
    return model2


def build_model5(X_train, y_train, X_valid, y_valid, max_len, max_features, embed_size, embedding_matrix, lr=0.0, lr_d=0.0, spatial_dr=0.0, dense_units=128, conv_size=128, dr=0.2, patience=3, fold_id=1):
    file_path = f"best_model_fold_{fold_id}.hdf5"
    check_point = ModelCheckpoint(
        file_path, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")
    early_stop = EarlyStopping(
        monitor="val_accuracy", mode="max", patience=patience)
    main_input = Input(shape=(max_len,), dtype='int32', name='main_input')
    x = (Embedding(max_features + 1, embed_size*2, input_length=max_len,
                   weights=[embedding_matrix], trainable=False))(main_input)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    hidden = concatenate([
        GlobalMaxPooling1D()(x),
        GlobalAveragePooling1D()(x),
        Attention(max_len)(x)
    ])
    hidden = Dense(256, activation='relu')(hidden)
    hidden = Dropout(0.5)(hidden)
    hidden = Dense(128, activation='relu')(hidden)
    hideen = Dropout(0.5)(hidden)
    output_lay = Dense(1, activation='sigmoid')(hidden)
    model = Model(inputs=[main_input], outputs=[output_lay])
    model.compile(loss="binary_crossentropy",
                  optimizer=Adam(), metrics=["accuracy"])
    model2 = Model(inputs=[main_input], outputs=[output_lay])
    model.fit(X_train, y_train, batch_size=16, epochs=20, validation_data=(
        X_valid, y_valid), verbose=1, callbacks=[early_stop, check_point])
    model2.load_weights(file_path)
    model2.compile(loss="binary_crossentropy",
                   optimizer=Adam(), metrics=["accuracy"])
    return model2
