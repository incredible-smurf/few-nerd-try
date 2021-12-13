import os
import matplotlib.pyplot as plt
def get_class_name(rawtag):
    # get (finegrained) class name
    if rawtag.startswith('B-') or rawtag.startswith('I-'):
        return rawtag[2:]
    else:
        return rawtag

class Sample():
    def __init__(self, filelines):
        filelines = [line.split('\t') for line in filelines]
        self.words, self.tags = zip(*filelines)
        self.words = [word.lower() for word in self.words]
        # strip B-, I-
        self.normalized_tags = list(map(get_class_name, self.tags))
        self.class_count = {}
        assert len(self.words)==len(self.tags)
        self.token_len=len(self.words)

    def __count_entities__(self):
        current_tag = self.normalized_tags[0]
        for tag in self.normalized_tags[1:]:
            if tag == current_tag:
                continue
            else:
                if current_tag != 'O':
                    if current_tag in self.class_count:
                        self.class_count[current_tag] += 1
                    else:
                        self.class_count[current_tag] = 1
                current_tag = tag
        if current_tag != 'O':
            if current_tag in self.class_count:
                self.class_count[current_tag] += 1
            else:
                self.class_count[current_tag] = 1

    def get_class_count(self):
        if self.class_count:
            return self.class_count
        else:
            self.__count_entities__()
            return self.class_count

    def get_tag_class(self):
        # strip 'B' 'I'
        tag_class = list(set(self.normalized_tags))
        if 'O' in tag_class:
            tag_class.remove('O')
        return tag_class

    def valid(self, target_classes):
        return (set(self.get_class_count().keys()).intersection(set(target_classes))) and not (set(self.get_class_count().keys()).difference(set(target_classes)))

    def __str__(self):
        newlines = zip(self.words, self.tags)
        return '\n'.join(['\t'.join(line) for line in newlines])


def load_data_from_file__( filepath):
        samples = []
        classes = []
        with open(filepath, 'r', encoding='utf-8')as f:
            lines = f.readlines()
        samplelines = []
        index = 0
        for line in lines:
            line = line.strip()
            if line:
                samplelines.append(line)
            else:
                sample = Sample(samplelines)
                samples.append(sample)
                sample_classes = sample.get_tag_class()
                #self.__insert_sample__(index, sample_classes)
                classes.append(sample_classes) 
                samplelines = []
                index += 1
        #classes = list(set(classes))
        return samples, classes



filepath='./data/few_ner/intra'
data_files= os.listdir(filepath)
for path in data_files:
    if '.txt' in path:
        path_now = os.path.join(filepath,path)
        samples ,classes =load_data_from_file__(path_now)
        maxlen=0
        maxclass_len=0
        assert len(classes)==len(samples)
        for i in range(len(samples)):
            if(samples[i].token_len>maxlen):
                maxlen=samples[i].token_len
            maxclass_len=max(maxclass_len,len(classes[i]))
        len_list=[0 for i in range(maxlen+1)]
        len_classes_num=[0 for i in range(maxclass_len+1)]
        for i in range(len(samples)):
            len_list[samples[i].token_len]+=1
            len_classes_num[len(classes[i])]+=1
            

        if not os.path.exists(os.path.join(filepath,'checking_image')):
            os.mkdir(os.path.join(filepath,'checking_image'))
        plt.plot([i for i in range(maxlen+1)],len_list)
        plt.xlabel('token length')
        plt.ylabel('token number')
        plt.title(path_now+' token lenth distribution')
        plt.savefig(os.path.join(filepath,'checking_image',path[:-4])+'_lenth_number.jpg')
        plt.show()
        plt.close()


        plt.plot([i for i in range(maxclass_len+1)],len_classes_num)
        plt.xlabel('class number')
        plt.ylabel('token number')
        plt.title(path_now+' class amount distribution')
        plt.savefig(os.path.join(filepath,'checking_image',path[:-4])+'_class_number.jpg')
        plt.show()
        plt.close()


