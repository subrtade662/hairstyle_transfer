from argparse import ArgumentParser
from unittest import result

from hairstyle_transfer_tool import Tool



def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-path", "--output_path", dest="path",required=True,
                    help="path to save the experiment")
    parser.add_argument("-mod", "--mode", dest="mode",required=True, choices=['transfer', 'interp', 'manip'],
                    help="name of the hairstyle source file, possible values: ['transfer', 'interp', 'manip']")
    parser.add_argument("-src", "--source", dest="source",required=True,
                    help="name of the hairstyle source file")
    parser.add_argument("-trg", "--target", dest="target",required=False,
                    help="name of the identity source file")
    
    parser.add_argument("-blend", "--alpha_blend", dest="alpha_blend",required=False, action='store_true',
                    help="alpha blending for better identity preservation")
    
    parser.add_argument("-nsteps", "--nsteps", dest="n_steps_interp",required=False, type=int,
                    help="number of steps for interpolation")
    
    parser.add_argument("-face_interp", "--interpolate_face", dest="face_interp",required=False, action='store_true',
                    help="Interpolation in the face domain (hair domain by default).")

    parser.add_argument("-attr", "--attribute", dest="manip_attribute",required=False, choices=['color', 'structure'],
                    help="which attribute to manipulate, possible values: ['color', 'structure']")
    parser.add_argument("-strength", "--strength", dest="manip_strength",required=False, type=float, nargs='+',
                    help="Manipulation strength")

    args = vars(parser.parse_args())
    check_args(args)
    return args


def check_args(args):
    print(args)
    if args['mode'] in ['transfer','interp']:
        assert args['target'] is not None, "Target image is mandatory for transfer and interp modes"

    if args['mode'] == 'interp':
        assert args['n_steps_interp'] is not None, 'You need to set --nsteps for interp mode.'
    if args['mode'] == 'manip':
        assert args['manip_attribute'] is not None, 'You need to set --attribute for manip mode.'
        assert args['manip_strength'] is not None, 'You need to set --strength for manip mode.'
    
    
    





def run_inference(tool, args):
    if args['mode'] == 'transfer':
        paths_to_results = tool.hairstyle_transfer(args['source'], args['target'], alpha_blend = args['alpha_blend'])
    elif args['mode'] == 'interp':
        paths_to_results = tool.interpolation_single_pair(args['source'], args['target'], n_steps=args['n_steps_interp'], interpolate_hair = not args['face_interp'], alpha_blend = args['alpha_blend'])
    elif args['mode'] == 'manip':
        paths_to_results = tool.hair_manipulation_single(args['source'], args['manip_attribute'], coeffs=args['manip_strength'], alpha_blend= args['alpha_blend'])
    else:
        raise 'Mode not recognized'
    return paths_to_results





if __name__ == '__main__':
    args = parse_args()
    print(f'All done')
    tool = Tool(opts=None, result_path='./data/results/', checkpoint_path='./best_model.pt')
    paths_to_results =run_inference(tool, args)
    print(f'Results saved as: {paths_to_results}')
