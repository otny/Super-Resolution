import argparse
import template


parser = argparse.ArgumentParser(description='EDSR and MDSR')

parser.add_argument('--debug', action='store_true',                     # デバックモードの有効化
                        help='Enables debug mode')
parser.add_argument('--template', default='.',                          # template.py を持ってくるかどうか
                        help='You can set various templates in option.py')


# Hardware specifications
parser.add_argument('--n_threads', type=int, default=2,
                        help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                        help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                        help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                        help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='./',               # データセットのディレクトリ
                        help='dataset directory')
parser.add_argument('--dir_demo', type=str, default='./demo_imgs',      # デモイメージを保存しておく場所？？？？
                        help='demo image directory')
parser.add_argument('--data_train', type=str, default='DIV2K',          # 訓練データセットの名前指定
                        help='train dataset name')
parser.add_argument('--data_test', type=str, default='Set5',                         # テストデータセットの名前指定
                        help='test dataset name')
parser.add_argument('--data_range', type=str, default='1-800/801-810',       # データ長？？？？
                        help='train/test data range')
parser.add_argument('--ext', type=str, default='sep',                   # データセットファイルの拡張？？？？？？　おｗ？？？
                        help='detaset file extention')
parser.add_argument('--scale', type=str, default='',                    # 超解像の倍率list : default='1.1 + 1.2 + 1.3 ....'とも書けるんご
                        help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=50,               # 出力のパッチサイズ？？？？？？
                        help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255,               # RGB の最大値？？？？？はぁ？？？
                        help='maximim value of RGB')
parser.add_argument('--n_colors', type=int, default=3,                    # 入出力の画像チャンネル数
                        help='number of color channels to use')
parser.add_argument('--chop', action='store_true',                      # メモリ効率の良い順伝播にする？？？？？？？？？
                        help='enable memory-efficient forward')
parser.add_argument('--no_augument', action='store_true',               # データ拡張を使わないか
                        help='do not use data augmentation')


# Moedl specifications
parser.add_argument('--model', default='MetaRDN',                          # モデル名
                        help='model name')
parser.add_argument('--act', type=str, default='relu',                  # 活性化関数名の指定
                        help='activation function')
parser.add_argument('--pre_train', type=str, default='.',               # 事前学習モデルのディレクトリ
                        help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',                  # 事前学習モデルの格拡張？？ディレクトリ？？？はぁああああ？？？？？
                        help='pre-trained model directory')             # なんで上と同じ help 文何だよクソ中国が！！！！！！！！！！！！
parser.add_argument('--n_resblocks', type=int, default=16,              # Residual Block の個数
                        help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,                  # 特徴マップのチャンネル数（個数）
                        help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,               # 残差スケール？？？？？？？
                        help='residual scaling')
parser.add_argument('--shift_mean', default=True,                       # 入力から pixel 平均を引く
                         help='subtract pixel mean from the input')
parser.add_argument('--dilation', action='store_true',                  # 拡張畳み込み（dilated conv）を使うかどうか
                        help='use dilated convolution')
parser.add_argument('--precision', type=str, default='single',          # 浮動小数点演算精度：単精度 | 半精度
                        choices=('single', 'half'),
                        help='FP precision for test (single | half)')


# Option for RDN : Residual dense network
parser.add_argument('--G0', type=int, default=64,                       # RDN で使う defaule のフィルター数
                        help='default number of filters. (Use in RDN)')
parser.add_argument('--RDNkSize', type=int, default=3,                  # RDN のkernel size
                        help='default kernel size. (Use in RDN)')
parser.add_argument('--RDNconfig', type=str, default='B',               # RDN のパラメータ設定
                        help='parameters config of RDN. (Use in RDN)')


# Option for RCAN : Residual channel attention network
parser.add_argument('--n_resgroups', type=int, default=10,              # 残差グループ数
                        help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,                # 特徴マップの削減
                        help='number of feature maps reduction')


# Training specifications (仕様)
parser.add_argument('--reset', action='store_true',                     # トレーニングをリセット
                        help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,             # テストを行うタイミング？（1000ごと）
                        help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=1000,                 # 総エポック数
                        help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,               # batch size
                        help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,               # 小分けにする？？？？？？
                        help='split the batch into smaller chunks')     # はああああ？？？？？？ なにいってんだこのクソ中国人がぁ！！
parser.add_argument('--self_ensemble', action='store_true',             # 自己集合法を用いて test するか
                        help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',
                        help='set this option to test the model')       # モデルのテスト
parser.add_argument('--gan_k', type=int, default=1,                     # Adversarial loss K
                        help='k value for adversarial loss')


# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,                   # 学習率
                        help='learning rate')
parser.add_argument('--lr_decay', type=int, default=200,                # 学習率減衰
                        help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step',           # 減衰タイプ
                        help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,                 # 学習率減衰係数
                        help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',                      # Optimizer
                        choices=('SGD', 'ADAM', 'RMSprop'),
                        help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,              # SGD のmomentum
                        help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9,                 # Adam beta1
                        help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,               # Adam beta2
                        help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,              # Adam epsilon
                        help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,            # 重み減衰
                        help='weight decay')
parser.add_argument('--start_epoch', type=int, default=0,               # スナップショットからの再開し始める epoch 数
                        help='resume from the snapshot, and the start_epoch')


# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',                 # 損失関数の構成
                        help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e6',      # 大きい誤差を持つスキップバッチ
                        help='skipping batch that has large error')


# Log specifications
parser.add_argument('--save', type=str, default='meta',                 # 保存ファイル名
                        help='file name to save')
parser.add_argument('--load', type=str, default='.',                    # ロードするファイル名（logs のディレクトリの下にあるファイル名を書く）
                        help='file name to load')
parser.add_argument('--resume', type=int, default=0,                    # 再開するチェックポイント (その int...epochのこと?????)
                        help='resume from specific checkpoint')
parser.add_argument('--save_models', action='store_true',               # 全ての中間モデルを保存
                        help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,             # トレーニングステータスをログに記録するタイミング
                        help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',              # 結果を保存
                        help='save output results')

args = parser.parse_args()
template.set_template(args)



# スケールの再定義
if args.scale=='':
        args.scale = [1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0]
else:
        args.scale = list(map(lambda x: float(x), args.scale.split('+')))
if args.epochs == 0:
        args.epochs = 1e8