# FullScenarioFloorplanParsing
This repository contains the official implementation of our paper "Intelligent Parsing of Floor Plans for Full-Scenario Applications". 


# 📖 Overview
This project provides a comprehensive solution for intelligent floor plan parsing, including:

​Preprocessing: Data preparation and augmentation for floor plan images

​Model Inference: Deep learning-based floor plan element detection and recognition

​Postprocessing: Refinement and structural analysis of parsing results

# 🚀 Quick Start
Prerequisites
```
# Clone the repository
git clone https://github.com/dididi0924/FullScenarioFloorplanParsing.git
cd FullScenarioFloorplanParsing

# uv 
uv venv --python 3.10
source .venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt
```

# 📥 Download Model Weights
Due to the large size of model files, please download them separately:

wall_mask.pt: 
https://koodi-1326775915.cos.ap-shanghai.myqcloud.com/wall_mask.onnx

# 🏃‍♂️ Usage Example


# Using Git LFS (For Developers)
If you have Git LFS installed:

```bash
git lfs install
git lfs pull
```


# 📝 Citation
If you use this code in your research, please cite our paper:

```bibtex
@article{author2024intelligent,
  title={Intelligent Parsing of Floor Plans for Full-Scenario Applications},
  author={Author, A. and Coauthor, B.},
  journal={Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```

# 📞 Contact
For questions and issues, please open a GitHub Issue or contact:

Email: xccd1234@163.com