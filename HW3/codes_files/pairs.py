from mrjob.job import MRJob
from mrjob.step import MRStep



class wordfrequency(MRJob):

    def configure_args(self):
        super(wordfrequency, self).configure_args()
        self.add_file_arg('--name', help='wikitext')

    def steps(self):
        return [
            MRStep(mapper=self.mapper_pairs(),
                   reducer=self.reducer_pairs())
        ]

    def mapper_pairs(self, key, value):
        for w in docs:
            for u in w.neighbours:
                yield [w,u], [key, value]

    def reducer_pairs(self, key, values):
        s = 0
        for c in counts:
            s = s + c

        yield [w, u], s


if __name__ == '__main__':
    wordfrequency.run()