from random import randrange
from csv import reader
from math import sqrt

class CustomKNN:
  def __init__(self):
    self.dataset = list()
    self.minmax = list()
    self.num_neighbors = 3
    self.n_folds = 5
    self.scores = list()

  # load a CSV file
  def load_data(self, filepath):
    with open(filepath, 'r') as file:
      csv_reader = reader(file)
      for row in csv_reader:
        if not row:
          continue
        self.dataset.append(row)
    return

  # test the knn on dataset
  def test(self):
    for i in range(len(self.dataset[0])-1):
      self.str_column_to_float(i)
    self.str_column_to_int(len(self.dataset[0])-1)

  # convert string column to float
  def str_column_to_float(self, column):
    for row in self.dataset:
      row[column] = float(row[column].strip())

  # convert string column to integer
  def str_column_to_int(self, column):
    class_values = [row[column] for row in self.dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
      lookup[value] = i
    for row in self.dataset:
      row[column] = lookup[row[column]]
    return lookup

  # find min and max values for each column
  def dataset_minmax(self):
    for i in range(len[self.dataset[0]]):
      col_values = [row[i] for row in self.dataset]
      value_min = min(col_values)
      value_max = max(col_values)
      self.minmax.append([value_min, value_max])
    return self.minmax

  # rescale dataset columns to the range 0-1
  def normalize_dataset(self):
    for row in self.dataset:
      for i in range(len(row)):
        row[i] = (row[i] - self.minmax[i][0]) / (self.minmax[i][1] - self.minmax[i][0])

  # split dataset into k folds
  def cross_validation_split(self):
    dataset_split = list()
    dataset_copy = list(self.dataset)
    fold_size = int(len(self.dataset) / self.n_folds)
    for _ in range(self.n_folds):
      fold = list()
      while len(fold) < fold_size:
        index = randrange(len(dataset_copy))
        fold.append(dataset_copy.pop(index))
      dataset_split.append(fold)
    return dataset_split

  # calculte accuracy percentage
  def accuracy_metric(self, actual, predicted):
    correct = 0
    for i in range(len(actual)):
      if actual[i] == predicted[i]:
        correct += 1
    return correct / float(len(actual)) * 100.0

  # evaluate an algorithm using a cross validation split
  def evaluate_algorithm(self):
    folds = self.cross_validation_split()
    for fold in folds:
      train_set = list(folds)
      train_set.remove(fold)
      train_set = sum(train_set, [])
      test_set = list()
      for row in fold:
        row_copy = list(row)
        test_set.append(row_copy)
        row_copy[-1] = None
      predicted = self.k_nearest_neighbors(train_set, test_set)
      actual = [row[-1] for row in fold]
      accuracy = self.accuracy_metric(actual, predicted)
      self.scores.append(accuracy)
    return self.scores

  # knn algorithm
  def k_nearest_neighbors(self, train, test):
    predictions = list()
    for row in test:
      output = self.predict_classification(row, train)
      predictions.append(output)
    return (predictions)

  # make a prediction with neighbors
  def predict_classification(self, test_row, train = None):
    if train is None:
      train = self.dataset
    neighbors = self.get_neighbors(train, test_row)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction

  # locate the most similar neighbors
  def get_neighbors(self, train, test_row):
    distances = list()
    for train_row in train:
      dist = self.euclidean_distance(test_row, train_row)
      distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(self.num_neighbors):
      neighbors.append(distances[i][0])
    return neighbors

  # calculate the euclidean distance between two vectors
  def euclidean_distance(self, row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
      distance += (row1[i] - row2[i])**2
    return sqrt(distance)

def main():

  model = CustomKNN()
  model.load_data('iris.csv')
  model.test()
  scores = model.evaluate_algorithm()

  print('Pontuação: %s' % scores)
  print('Precisão média: %.3f%%' % (sum(scores)/float(len(scores))))

  print('\n\n')
  print('Label           | value\n')
  print('-----------------------\n')
  print('Iris-virginica  |  0\n')
  print('Iris-setosa     |  1\n')
  print('Iris-versicolor |  2\n')

  sample_rows = [
    [5.7,2.9,4.2,1.3],
    [4.7,3.1,1.3,0.2],
    [6.5,3.3,5.3,2.3]
  ]

  for row in sample_rows:
    label = model.predict_classification(row)
    print('Exemplo = %s, Predição: %s' % (row, label))

main();