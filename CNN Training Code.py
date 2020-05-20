#  read in training data
dataframe = pd.read_csv('labels_new.csv')

#set image size and number of classes being classied
NUM_CLASSES = 2
IMG_WIDTH = 90
IMG_HEIGHT = 335

# selected_docs is an array holding the number of classes wanted to train and test on (in this case both - 2 classes)
selected_docs = list(dataframe.groupby('aligned').count().sort_values(by='id', ascending=False).head(NUM_CLASSES).index)

# df_sub_train is filled with the records that are associated with the two classes held in selected_docs.
df_sub_train = dataframe[dataframe['aligned'].isin(selected_docs)]

# targets is a series object. A series object is like a mix between a dataframe and an array. In other words it is
# an array with axis labels or titles.
targets = pd.Series(df_sub_train['aligned'])

# one_hot is a dataframe where the shred pairs class labels have been one hot encoded.
# Whether it is a match or not is given a unique label that can be used to easily identify a match.
one_hot = pd.get_dummies(targets, sparse=True)

# one_hot_labels uses the data from the one_hot dataframe to create a 2D
# array where each array is an identifier for whether the doc is a match or not.
one_hot_labels = np.asarray(one_hot)

# One Hot Encode the classes
data = np.array(selected_docs)
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded = onehot_encoder.fit_transform(data.reshape(-1,1))

# Function to read an image and resize it accordingly
def read_img(img_id):
    img = cv2.imread(img_id)
    return cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

#  set up images, labels and classes array's
images = []
image_ids = []
classes = []

# Load all the images/image ids/class labels and append them in appropiate array
for img_id, aligned in tqdm.tqdm(df_sub_train.values):
      #append remaining images with ID tags to appropiate arrays
      images.append(read_img(img_id))
      image_ids.append(img_id)
      classes.append(aligned)

# Train test split data

#  Split data into train and test
# Let's store the image ids as X and the one hot labels as Y
X = np.array(image_ids)
Y = np.array(one_hot_labels)

# Initially split the data into train and test/val arrays
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, shuffle=True)


# After splitting the data in their individual arrays, you can load the images
# for each of the image ids stored in the x_train/x_test/x_val variables
x_train_images = []
x_test_images = []

#  append labels to train/test images

for i in tqdm.tqdm(range(0, len(x_train))):
    x_train_images.append(read_img(x_train[i]))
x_train_images = np.array(x_train_images)

for i in tqdm.tqdm(range(0, len(x_test))):
    x_test_images.append(read_img(x_test[i]))
x_test_images = np.array(x_test_images)

# Get the CNN model without it's fully connected layer and load no pre trained weights
bottleneck = Xception(include_top=False,
                             weights=None,
                             input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                             classes=NUM_CLASSES)

# Add the fully connected layer we need to output the number of classes
model = bottleneck.output
model = Flatten()(model)
model = Dense(256, activation='sigmoid')(model)
model = Dropout(0.5)(model)
model = Dense(64, activation='sigmoid')(model)

predictions = Dense(NUM_CLASSES, activation='sigmoid')(model)

final_model = Model(inputs=[bottleneck.input], outputs=[predictions])

# Compile the model and train it for optimum epcohs. Also insert the validation_data to make sure we don't overfit the model
final_model.compile(SGD(lr=0.1, momentum=0.9), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
history = final_model.fit(x_train_images, y_train, epochs=10, batch_size=64, validation_data=(x_test_images, y_test))

#output vaildation accuracy
scores = final_model.evaluate(x_test_images, y_test, verbose=0)
print("%s: %.2f%%" % (final_model.metrics_names[1], scores[1]*100))

#save the train CNN
final_model.save("Xception_history_10_epochs.h5")

#Output the loss, validation accuracy, categorical accuracy and associated losses for each epoch of training
pd.DataFrame(history.history).to_csv("history_Xception.csv")
