class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'jingwei':
            return '/data/dingyifeng/jingwei/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
