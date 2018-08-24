from keras.models import model_from_json


def load_cnn(model_json_path, model_h5_path):

    json_file = open(model_json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    cnn = model_from_json(loaded_model_json)
    # load weights into new model
    cnn.load_weights(model_h5_path)
    print("Loaded model from disk")

    opt = 'adam'
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']

    # Compile the classifier using the configuration we want
    cnn.compile(optimizer=opt, loss=loss, metrics=metrics)

    return cnn

