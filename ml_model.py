#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 03:24:52 2025

Created by: Dhananjoy Bhuyan

ML Model
"""

from random import choice

"""
My own kind of model here...
How my model works:
    it just does:
        y = x <operation> <param>
    where operation is an arithmetic operation and param is a number that gets operated with x.
    
So it tries different operations, it adjusts param accordingly based on different operations and Xs.

Then it compares the error between all the operations and keeps the operation and param with the least error.

"""

# Model class definition


class ArithmeticModel:
    def __init__(self):
        self.param = 1  # initialize param to 1.

        self.operation = '+'  # operation to apply.

    def predict(self, x: float) -> float:
        # just do some math and return the result.
        return eval(f'{x} {self.operation} {self.param}')

    def __train_op(self, data: dict[float, float]) -> None:
        # adjust param based on data and operation.
        # take params for the data and average the param to fit the line through the middle, linear regression!
        if self.operation == '+':
            self.param = [y - x for x, y in data.items()]
        elif self.operation == '-':
            self.param = [x - y for x, y in data.items()]
        elif self.operation == '*':
            self.param = [y/x for x, y in data.items() if x != 0]
        else:
            self.param = [x/y for x, y in data.items() if y != 0]

        self.param = sum(self.param) / len(self.param) if self.param else 0

    def train(self, data: dict[float, float]) -> None:
        if len(data) < 4:
            raise ValueError('I need more data, this is so less......')
        mean_diffs = {}  # to store the avg difference for each operation.
        training_set = dict(list(data.items())[:len(data)//2 + 1])
        test_set = dict(list(data.items())[len(data)//2 + 1:])
        for op in '/*-+':
            self.operation = op  # set operation.
            self.__train_op(training_set)  # train the model.
            # to store the percentage differences accross the data.
            percentage_diffs = []
            for x in test_set:
                pred = self.predict(x)  # predicted.
                exp = test_set[x]  # expected.
                # the difference.
                percentage_diffs.append((abs(pred - exp)/exp)*100)
            # append the mean(avg) of the differences.
            mean_diffs[sum(percentage_diffs)/len(percentage_diffs)] = op
        # the best operation.
        self.operation = mean_diffs[min(list(mean_diffs.keys()))]
        self.__train_op(data)  # train on the best operation.


"""
Another kind of model, which uses the the conventional  mx + c formula to give output.... but but but... it won't use the gradient decend or something complex, we will use Grade 7 algebra instead. And I give you garantee that it will give output same or better than gradient decend or anything and it can be trained perfectly on less data. Better output on less data!! HAHA!!!!
But it will only work for fully linear data. And may not give really good results for slightly non-linear data.
"""

# Model 2 class definition.

# Works only for fully linear data set.


class AlgebricModel:
    def __init__(self):
        # random guess for parameters.
        self.params = {
            'm': 1,
            'c': 1
        }

    # prediction function.
    def predict(self, x: float) -> float:
        # return mx + c
        return self.params['m'] * x + self.params['c']

    # Error if user doesn't give enough data(means he is a fool).

    class NotEnoughDataError(Exception):
        def __init__(self, msg: str = ''):
            super().__init__(msg)

    # train function.
    def train(self, data: dict[float, float]) -> None:
        # make sure not to modify the real data in-place.
        d = dict(list(data.items())[:])

        # check if data is too less.
        if len(d) < 2:
            raise self.NotEnoughDataError(
                "I need more data. This is too less.")

        # if 0 is in data, so we can easily get c, because when x=0, y = c.
        if 0 in d:
            self.params['c'] = d[0]  # Yay! We have c.
            d.pop(0)  # we don't want 0 division error.
            rand_key = choice(list(d.keys()))  # take any key that's not 0.
            # calculate m as (y - c)/x. ALGEBRA!!
            self.params['m'] = (d[rand_key] - self.params['c'])/rand_key
        else:  # No x as 0 in data?????????? FINE!
            rand_key1 = list(d.keys())[0]  # first key.
            val1 = d[rand_key1]  # value.
            rand_key2 = list(d.keys())[1]  # second key.
            val2 = d[rand_key2]  # second value.

            # we calculate slope m, as dy/dx (means change in y by change in x).
            # so it is basically:
            # (y2 - y1)/(x2 - x1)
            # the coordinates ending with 2 are greater than the ones ending with 1, so y2 > y1 and x2 > x1. So we take max(y, y) - min(x, x) so whichever value is greater it will be the first number.
            self.params['m'] = (max(val2, val1) - min(val2, val1)) / \
                (max(rand_key2, rand_key1) - min(rand_key2, rand_key1))

            # so now c is just y - mx!! ALGEBRAAA!!! Again!
            self.params['c'] = val1 - (self.params['m'] * rand_key1)


"""
One more model, this time with gradient decend!!
"""

# model class definition


class GradientDecendModel:
    def __init__(self):
        # initial guess.
        self.params = {
            'm': 1,
            'c': 1
        }

    # predict function.
    def predict(self, x: float) -> float:
        # just mx + c yay!
        return self.params['m']*x + self.params['c']

    # the hard part... Training function.

    def train(self, data: dict[float, float]) -> None:
        # dynamic learning rate so that values don't explode and become NaN.
        self.learning_rate = float(
            # Add  digits number of zeros of the greatest number in the data.
            '0.' + '0'*len(str(int(max(list(data.values()))))) + '1')

        # keep it below the maximum limit.
        if self.learning_rate > 0.01:
            self.learning_rate = 0.01

        prev_avg_error = None  # the error in the previous iteration.
        # how many times there was no improvement in the error.
        same_error_count = 0
        # iterate.
        while 1:

            avg_error = []  # store the errors first.

            # mark previous params.
            prevm = self.params['m']
            prevc = self.params['c']

            # train.
            for x, y in data.items():
                pred = self.predict(x)  # predict.
                error = y - pred  # calculate error.
                # so that error is not negative, it's for getting the real avg.
                avg_error.append(abs(error))

                # direction * aggression * scale.
                self.params['m'] += error * self.learning_rate * x
                # aggression * direction.
                self.params['c'] += self.learning_rate * error
            # All errors stored, now calculate the avg.
            avg_error = sum(avg_error)/len(avg_error)
            if prev_avg_error:  # if that's not none.

                # if error increased or 10+ times the error was same.
                if (avg_error > prev_avg_error) or (same_error_count > 10):
                    # previous params we're good boys.
                    self.params['m'] = prevm
                    self.params['c'] = prevc
                    # no need to iterate anymore.
                    break
                if avg_error == prev_avg_error:
                    # if error's same count it.
                    same_error_count += 1
            # update previous error.
            prev_avg_error = avg_error


# test them all.
def main() -> list[str]:
    house_data = {
        600: 1800000,
        800: 2400000,
        1000: 3100000,
        1200: 3750000,
        1500: 4650000,
        1800: 5400000,
        2000: 6300000,
        2200: 7150000,
        2500: 8000000,
        3000: 9600000
    }

    test_data = {
        3200: 10200000,
        3500: 11300000,
        3800: 12450000,
        4000: 13400000,
        4500: 15000000,
        5000: 16600000,
        5500: 18200000,
        6000: 19800000,
        7000: 23100000,
        8000: 26500000,
        10000: 33000000
    }

    m1 = ArithmeticModel()
    m2 = AlgebricModel()
    m3 = GradientDecendModel()

    models = [m1, m2, m3]

    for m in models:
        m.train(house_data)
    model_tests = []
    for m in models:
        percentage_diffs = []
        for x, y in test_data.items():
            pred = m.predict(x)
            percentage_diffs.append(((abs(pred - y)/y)*100))

        model_tests.append(f'{min(percentage_diffs)}-{max(percentage_diffs)}') # error percentage range.
    return model_tests


if __name__ == '__main__':
    main()
