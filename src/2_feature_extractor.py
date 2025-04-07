from deepface import DeepFace
import os
import pickle
from tqdm import tqdm
from src.utils.all_utils import read_yaml, create_directory

def feature_extractor(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    artifacts = config['artifacts']
    artifacts_dir = artifacts['artifacts_dir']
    pickle_format_data_dir = artifacts['pickle_format_data_dir']
    img_pickle_file_name = artifacts['img_pickle_file_name']
    img_pickle_file_name = os.path.join(artifacts_dir, pickle_format_data_dir, img_pickle_file_name)
    filenames = pickle.load(open(img_pickle_file_name, 'rb'))

    # Directory for extracted features
    feature_extraction_path = os.path.join(artifacts_dir, artifacts['feature_extraction_dir'])
    create_directory(dirs=[feature_extraction_path])

    feature_name = os.path.join(feature_extraction_path, artifacts['extracted_features_name'])
    features = []

    # Use DeepFace for feature extraction
    for file in tqdm(filenames):
        try:
            embeddings = DeepFace.represent(img_path=file, model_name='VGG-Face', enforce_detection=False)
            if embeddings:
                features.append(embeddings[0]["embedding"])
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

    # Save extracted features
    pickle.dump(features, open(feature_name, 'wb'))
    print(f"Features saved to {feature_name}")

if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--config', "-c", default='config/config.yaml')
    args.add_argument('--params', "-p", default='params.yaml')
    parsed_args = args.parse_args()
    feature_extractor(config_path=parsed_args.config, params_path=parsed_args.params)
