import  kaggle.representation_learning.Transformer.TransformerImplementations.all as alltransform
import kaggle.representation_learning.Transformer.TransformerImplementations.categorical as cattransform
import kaggle.representation_learning.Transformer.TransformerImplementations.numeric as numtransform
from kaggle.representation_learning.Transformer.TransformerImplementations.parser.DateTransformer import DateTransformer
import kaggle.representation_learning.Transformer.TransformerImplementations.binary as bintransform

def get_all_transformations_per_column(pandas_table, target_column):
    numerical_transformers = [numtransform.BinarizerTransformer(0),
                              numtransform.BucketTransformer(0),
                              numtransform.ClusterDistTransformer(0),
                              numtransform.ClusterTransformer(0),
                              numtransform.CosTransformer(0),
                              numtransform.DegreesTransformer(0),
                              numtransform.DummyTransformer(0),
                              numtransform.IdentityTransformer(0),
                              numtransform.ImputerTransformer(0),
                              numtransform.LogTransformer(0),
                              numtransform.PlottingPositionTransformer(0),
                              numtransform.PolynomialTransformer(0),
                              numtransform.QuantileTransformer(0),
                              numtransform.RadiansTransformer(0),
                              numtransform.RankTransformer(0),
                              numtransform.RSHTransformer(0),
                              numtransform.ScaleTransformer(0),
                              numtransform.SigmoidTransformer(0),
                              numtransform.SinTransformer(0),
                              numtransform.SqrtTransformer(0),
                              numtransform.SquareTransformer(0),
                              numtransform.TanTransformer(0),
                              numtransform.ToIntTransformer(0),
                              numtransform.TrimtailTransformer(0),
                              numtransform.WinsorizeTransformer(0),
                              numtransform.ZScoreTransformer(0)]
    categorical_transformers = [cattransform.OneHotTransformer(0),
                                cattransform.FrequencyEncodingTransformer(0),
                                cattransform.OrdinalTransformer(0)]
    all_transformers = [  # alltransform.AvgWord2VecTransformer(0),
        alltransform.HashingTransformer(0),
        alltransform.LengthCountTransformer(0),
        alltransform.NgramTransformer(0, analyzer='char'),
        alltransform.NgramTransformer(0, analyzer='word'),

        alltransform.ParseNumbersTransformer(0),
        DateTransformer(0)
    ]

    transformations = {}

    for col_rep in range(pandas_table.shape[1]):

        if col_rep != target_column:

            transformations[col_rep]= []
            transformations[col_rep].extend(all_transformers)

            # check if numerical
            if pandas_table.dtypes[col_rep] == 'int64' or pandas_table.dtypes[col_rep] == 'float64':
                transformations[col_rep].extend(numerical_transformers)

            # check if categorical
            count = len(pandas_table[pandas_table.columns[col_rep]].unique())
            fraction = count / float(len(pandas_table))
            if fraction < 0.05:
                transformations[col_rep].extend(categorical_transformers)

    return transformations


def get_binary_transformations():
    binary_transformations = [bintransform.Addition(0,0)]
    return binary_transformations




def get_unary_transformations():
    numerical_transformers = [numtransform.BinarizerTransformer(0),
                              #numtransform.BucketTransformer(0),
                              #numtransform.ClusterDistTransformer(0),
                              #numtransform.ClusterTransformer(0),
                              #numtransform.CosTransformer(0),
                              #numtransform.DegreesTransformer(0),
                              #numtransform.DummyTransformer(0),
                              numtransform.IdentityTransformer(0),
                              #numtransform.ImputerTransformer(0),
                              numtransform.LogTransformer(0),
                              #numtransform.PlottingPositionTransformer(0),
                              #numtransform.PolynomialTransformer(0),
                              #numtransform.QuantileTransformer(0),
                              #numtransform.RadiansTransformer(0),
                              #numtransform.RankTransformer(0),
                              #numtransform.RSHTransformer(0),
                              #numtransform.ScaleTransformer(0),
                              #numtransform.SigmoidTransformer(0),
                              #numtransform.SinTransformer(0),
                              #numtransform.SqrtTransformer(0),
                              #numtransform.SquareTransformer(0),
                              #numtransform.TanTransformer(0),
                              #numtransform.ToIntTransformer(0),
                              #numtransform.TrimtailTransformer(0),
                              #numtransform.WinsorizeTransformer(0),
                              #numtransform.ZScoreTransformer(0)
                              ]
    categorical_transformers = [#cattransform.OneHotTransformer(0),
                                #cattransform.FrequencyEncodingTransformer(0),
                                #cattransform.OrdinalTransformer(0)
                               ]
    all_transformers = [  # alltransform.AvgWord2VecTransformer(0),
        #alltransform.HashingTransformer(0),
        #alltransform.LengthCountTransformer(0),
        #alltransform.NgramTransformer(0, analyzer='char'),
        #alltransform.NgramTransformer(0, analyzer='word'),

        #alltransform.ParseNumbersTransformer(0),
        #DateTransformer(0)
    ]

    transformations = []


    transformations.extend(all_transformers)
    transformations.extend(numerical_transformers)
    transformations.extend(categorical_transformers)

    return transformations











def get_all_transformations():
    numerical_transformers = [numtransform.BinarizerTransformer(0),
                              numtransform.BucketTransformer(0),
                              numtransform.ClusterDistTransformer(0),
                              numtransform.ClusterTransformer(0),
                              numtransform.CosTransformer(0),
                              numtransform.DegreesTransformer(0),
                              numtransform.DummyTransformer(0),
                              numtransform.IdentityTransformer(0),
                              numtransform.ImputerTransformer(0),
                              numtransform.LogTransformer(0),
                              numtransform.PlottingPositionTransformer(0),
                              numtransform.PolynomialTransformer(0),
                              numtransform.QuantileTransformer(0),
                              numtransform.RadiansTransformer(0),
                              numtransform.RankTransformer(0),
                              numtransform.RSHTransformer(0),
                              numtransform.ScaleTransformer(0),
                              numtransform.SigmoidTransformer(0),
                              numtransform.SinTransformer(0),
                              numtransform.SqrtTransformer(0),
                              numtransform.SquareTransformer(0),
                              numtransform.TanTransformer(0),
                              numtransform.ToIntTransformer(0),
                              numtransform.TrimtailTransformer(0),
                              numtransform.WinsorizeTransformer(0),
                              numtransform.ZScoreTransformer(0)]
    categorical_transformers = [cattransform.OneHotTransformer(0),
                                cattransform.FrequencyEncodingTransformer(0),
                                cattransform.OrdinalTransformer(0)]
    all_transformers = [  # alltransform.AvgWord2VecTransformer(0),
        alltransform.HashingTransformer(0),
        alltransform.LengthCountTransformer(0),
        alltransform.NgramTransformer(0, analyzer='char'),
        alltransform.NgramTransformer(0, analyzer='word'),

        alltransform.ParseNumbersTransformer(0),
        DateTransformer(0)
    ]

    transformations = []


    transformations.extend(all_transformers)
    transformations.extend(numerical_transformers)
    transformations.extend(categorical_transformers)

    return transformations