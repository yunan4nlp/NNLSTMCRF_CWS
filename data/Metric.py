class Metric:
    def __init__(self):
        self.overall_label_count = 0
        self.correct_label_count = 0
        self.predicated_label_count = 0

    def reset(self):
        self.overall_label_count = 0
        self.correct_label_count = 0
        self.predicated_label_count = 0

    def bIdentical(self):
        if self.predicated_label_count == 0:
            if self.overall_label_count == self.correct_label_count:
                return True
            return False
        else:
            if self.overall_label_count == self.correct_label_count and \
                    self.predicated_label_count == self.correct_label_count:
                return True
            return False

    def getAccuracy(self):
        if self.overall_label_count + self.predicated_label_count == 0:
            return 1.0
        if self.predicated_label_count == 0:
            return self.correct_label_count*1.0 / self.overall_label_count
        else:
            return self.correct_label_count*2.0 / (self.overall_label_count + self.predicated_label_count)

    def print(self):
        if self.predicated_label_count == 0:
            print("Accuracy:\tP=" + str(self.correct_label_count) + '/' + str(self.overall_label_count))
        else:
            print("Recall:\tP=" + str(self.correct_label_count) + "/" + str(self.overall_label_count) + "=" + str(self.correct_label_count*1.0 / self.overall_label_count), end=",\t")
            print("Accuracy:\tP=" + str(self.correct_label_count) + "/" + str(self.predicated_label_count) + "=" + str(self.correct_label_count*1.0 / self.predicated_label_count), end=",\t")
            print("Fmeasure:\t" + str(self.correct_label_count*2.0 / (self.overall_label_count + self.predicated_label_count)))
