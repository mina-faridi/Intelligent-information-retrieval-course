from mrjob.job import MRJob
from mrjob.step import MRStep

class wordfrequency(MRJob):

    def configure_args(self):
        super(wordfrequency, self).configure_args()
        self.add_file_arg('--name', help='wikitext')

    def steps(self):
        return [
            MRStep(
                mapper=self.mapper_stripes(),
                reducer=self.reducer_stripes())
        ]

    def mapper_stripes(self, _, line):
        for w in docs:
            H = []
            for u in w.neighbours:
                H[u]= H[u]+1
                yield w, H

    def reducer_stripes(self, key, values):
        Hf = []
        for H in Hs:
            sum(Hf, H)

        yield w, Hf





if __name__ == '__main__':
    wordfrequency.run()