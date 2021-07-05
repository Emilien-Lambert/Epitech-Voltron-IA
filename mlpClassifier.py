from sklearn.neural_network import MLPClassifier


def classifier(train_images, train_labels, test_images, test_labels):
    print("\nStarting MLP Classifier...... \n")
    clf = MLPClassifier(verbose=True, hidden_layer_sizes=(5000, 3))
    clf.fit(train_images, train_labels)
    print(clf.predict(test_images))
    print(clf.score(test_images, test_labels))
