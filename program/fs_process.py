from feature_selection.filter_selection import FilterSelection

class FSProcess():
    def __init__(self, d1_x, d1_y, d2_x, d2_y):
        self.d1x = d1_x
        self.d1y = d1_y
        self.d2x = d2_x
        self.d2y = d2_y

    # Output features selected by method
    def display_features(self, model, d1, d2):
        print(f'{model} Dataset 1 Features: {d1.columns}')
        print(f'{model} Dataset 2 Features: {d2.columns}')

    # Create function based switch method for selecting fs method
    def exec_fs(self, opt):
        fs = FilterSelection(7)

        if opt < 1 or opt > 4:
            print(f'\n{opt} is not a valid option\n\n')
            return None, None

        if opt == 1:
            d1 = fs.chi2(self.d1x, self.d1y)
            d2 = fs.chi2(self.d2x, self.d2y)
            method = 'Chi Square'

        if opt == 2:
            d1 = fs.mi(self.d1x, self.d1y)
            d2 = fs.mi(self.d2x, self.d2y)
            method = 'Mutual Information'

        if opt == 3:
            d1 = fs.mrmr(self.d1x)
            d2 = fs.mrmr(self.d2x)
            method = 'Max Relevance Min Redundancy'

        if opt == 4:
            d1 = fs.rf(self.d1x, self.d1y)
            d2 = fs.rf(self.d2x, self.d2y)
            method = 'ReliefF'


        self.display_features(method, d1, d2)
        return d1, d2
