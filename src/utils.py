class Grim(object):
    def __init__(
        self,
        template_type,
        subtype,
        text,
        hypo1,
        hypo2,
        name1,
        name2,
        target,
        gold="neutral",
    ):
        """
        NLI data pair
        """
        self.template_type = template_type
        self.subtype = subtype
        self.text = text
        self.hypo1 = hypo1
        self.hypo2 = hypo2
        self.name1 = name1.name
        self.gender1 = name1.gender
        self.race1 = name1.race
        self.name2 = name2.name
        self.gender2 = name2.gender
        self.race2 = name2.race
        self.target = target
        self.gold = gold
        self.score1 = None
        self.score2 = None
        self.pred1 = None
        self.pred2 = None
        self.acc = None
        self.match = None
        self.net_diff = None

    def __get_pred(self, score):
        max_score = 0
        for key in score.keys():
            if max_score < score[key]:
                max_score = score[key]
                pred = key
        return pred

    def evaluate_pair(self):
        acc = 0
        match = 0
        if self.pred1 == self.gold:
            acc += 1
        if self.pred2 == self.gold:
            acc += 1
        if self.pred1 == self.pred2:
            match = 1
        net_diff = abs(self.score1[self.gold] - self.score2[self.gold])
        self.acc = acc/2
        self.match = match
        self.net_diff = net_diff
        return

    def generate_pair(self):
        sent1 = self.text + "[SEP]" + self.hypo1
        sent2 = self.text + "[SEP]" + self.hypo2

        return [sent1, sent2]

    def get_score(self, model):
        #  output1 = model(f"{self.text}[SEP]{self.hypo1}")
        #  self.score1 = dict()
        #  for out in output1[0]:
        #      self.score1[out['label']] = out['score']
        #  output2 = model(f"{self.text}[SEP]{self.hypo2}")
        #  self.score2 = dict()
        #  for out in output2[0]:
        #      self.score2[out['label']] = out['score']

        sent_pair = self.generate_pair()
        output = model(sent_pair)
        self.score1 = dict()
        self.score2 = dict()
        for out in output[0]:
            self.score1[out["label"]] = out["score"]
        for out in output[1]:
            self.score2[out["label"]] = out["score"]

        self.pred1 = self.__get_pred(self.score1)
        self.pred2 = self.__get_pred(self.score2)
        return


class Name(object):
    def __init__(self, name, gender, race, count=-1):
        """
        Name data
        """
        self.name = name
        self.gender = gender
        self.race = race
        self.count = count
