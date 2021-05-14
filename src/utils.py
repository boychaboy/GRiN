import numpy as np


class Grin(object):
    def __init__(
        self,
        template_type,
        subtype,
        text,
        hypo1,
        hypo2,
        hypo3=None,
        hypo4=None,
        name1=None,
        name2=None,
        name3=None,
        name4=None,
        target=None,
        unrelated=None,
        text2=None,
        gold=None,
    ):
        """
        NLI data pair
        """
        self.template_type = template_type
        self.subtype = subtype
        self.text = text
        self.hypo1 = hypo1
        self.hypo2 = hypo2
        self.nn_cnt = 2
        if unrelated:
            self.unrelated = unrelated
            self.nn_cnt = 3
        else:
            self.unrelated = None
        if text2:
            self.text2 = text2
        else:
            self.text2 = None
        if type(name1) is Name:
            self.name1 = name1.name
            self.gender1 = name1.gender
            self.race1 = name1.race
        if type(name2) is Name:
            self.name2 = name2.name
            self.gender2 = name2.gender
            self.race2 = name2.race
        if hypo3:
            self.hypo3 = hypo3
        if hypo4:
            self.hypo4 = hypo4
            self.nn_cnt = 4
        self.target = target
        self.score1 = None
        self.score2 = None
        self.score3 = None
        self.score4 = None
        self.gold = gold
        self.pred1 = None
        self.pred2 = None
        self.pred3 = None
        self.pred4 = None
        self.acc = None
        self.nn1 = None
        self.nn2 = None
        self.nn3 = None
        self.nn4 = None
        self.nn_avg = None
        self.nn_dif = None
        self.nn_std = None
        self.lms = None
        self.ss = None

    def __get_pred(self, score):
        max_score = 0
        for key in score.keys():
            if max_score < score[key]:
                max_score = score[key]
                pred = key
        return pred

    def evaluate_pair(self):
        acc = 0
        self.entail = list(self.score1.keys())[0]  # entail label
        self.gold = list(self.score1.keys())[1]  # neutral label
        self.contra = list(self.score1.keys())[2]  # contradict label

        if self.template_type == 'C':
            if self.pred1 == self.gold:
                acc += 1
            if self.pred2 == self.gold:
                acc += 1
            if int(self.subtype) > 2:  # StereoSet
                if self.pred3 == self.gold:
                    acc += 1
            self.nn1 = self.score1[self.gold]
            self.nn2 = self.score2[self.gold]
            self.en1 = self.score1[self.entail]
            self.en2 = self.score2[self.entail]
            self.cn1 = self.score1[self.contra]
            self.cn2 = self.score2[self.contra]

            en = np.array([self.en1, self.en2])
            nn = np.array([self.nn1, self.nn2])
            cn = np.array([self.cn1, self.cn2])

            #  enc1 = list(self.score1.values())
            #  enc2 = list(self.score2.values())
            #  score_vec = np.array([enc1, enc2])

        else:  # type A, B
            if int(self.subtype) < 3:  # gender pair
                if self.pred1 == self.gold:
                    acc += 1
                if self.pred2 == self.gold:
                    acc += 1
                self.nn1 = self.score1[self.gold]
                self.nn2 = self.score2[self.gold]
                self.en1 = self.score1[self.entail]
                self.en2 = self.score2[self.entail]
                self.cn1 = self.score1[self.contra]
                self.cn2 = self.score2[self.contra]

                en = np.array([self.en1, self.en2])
                nn = np.array([self.nn1, self.nn2])
                cn = np.array([self.cn1, self.cn2])

                #  enc1 = list(self.score1.values())
                #  enc2 = list(self.score2.values())
                #  score_vec = np.array([enc1, enc2])
            else:
                if self.pred1 == self.gold:
                    acc += 1
                if self.pred2 == self.gold:
                    acc += 1
                if self.pred3 == self.gold:
                    acc += 1
                if self.pred4 == self.gold:
                    acc += 1
                self.nn1 = self.score1[self.gold]
                self.nn2 = self.score2[self.gold]
                self.nn3 = self.score3[self.gold]
                self.nn4 = self.score4[self.gold]
                self.en1 = self.score1[self.entail]
                self.en2 = self.score2[self.entail]
                self.en3 = self.score3[self.entail]
                self.en4 = self.score4[self.entail]
                self.cn1 = self.score1[self.contra]
                self.cn2 = self.score2[self.contra]
                self.cn3 = self.score3[self.contra]
                self.cn4 = self.score4[self.contra]

                en = np.array([self.en1, self.en2, self.en3, self.en4])
                nn = np.array([self.nn1, self.nn2, self.nn3, self.nn4])
                cn = np.array([self.cn1, self.cn2, self.cn3, self.cn4])

                #  enc1 = list(self.score1.values())
                #  enc2 = list(self.score2.values())
                #  enc3 = list(self.score3.values())
                #  enc4 = list(self.score4.values())
                #  score_vec = np.array([enc1, enc2, enc3, enc4])

        self.acc = acc / self.nn_cnt
        self.nn_avg = np.mean(nn)
        self.en_avg = np.mean(en)
        self.cn_avg = np.mean(cn)
        self.nn_std = np.std(nn)
        self.en_std = np.std(en)
        self.cn_std = np.std(cn)

        #  self.std = np.std(score_vec)

        return

    def generate_pair(self):
        if self.text2:
            # cp
            sent1 = self.text + "[SEP]" + self.hypo1
            sent2 = self.text2 + "[SEP]" + self.hypo2
            return [sent1, sent2]

        elif self.unrelated:
            # ss
            sent1 = self.text + "[SEP]" + self.hypo1
            sent2 = self.text + "[SEP]" + self.hypo2
            sent3 = self.text + "[SEP]" + self.unrelated
            return [sent1, sent2, sent3]

        elif self.nn_cnt == 4:
            # race
            sent1 = self.text + "[SEP]" + self.hypo1
            sent2 = self.text + "[SEP]" + self.hypo2
            sent3 = self.text + "[SEP]" + self.hypo3
            sent4 = self.text + "[SEP]" + self.hypo4
            return [sent1, sent2, sent3, sent4]

        else:
            # gender
            sent1 = self.text + "[SEP]" + self.hypo1
            sent2 = self.text + "[SEP]" + self.hypo2
            return [sent1, sent2]

    def get_score(self, model):

        sent_pair = self.generate_pair()
        output = model(sent_pair)

        if len(sent_pair) >= 2:
            self.score1 = dict()
            self.score2 = dict()
            for out in output[0]:
                self.score1[out["label"]] = out["score"]
            for out in output[1]:
                self.score2[out["label"]] = out["score"]
            self.pred1 = self.__get_pred(self.score1)
            self.pred2 = self.__get_pred(self.score2)

        if len(sent_pair) >= 3:
            self.score3 = dict()
            for out in output[2]:
                self.score3[out["label"]] = out["score"]
            self.pred3 = self.__get_pred(self.score3)

        if len(sent_pair) == 4:
            self.score4 = dict()
            for out in output[3]:
                self.score4[out["label"]] = out["score"]
            self.pred4 = self.__get_pred(self.score4)

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
