# SIN-Project
# Disease-Protein Association Prediction

This project implements and evaluates multiple algorithms for predicting disease-protein associations, with a specific focus on Liver Carcinoma. The implementation includes various network-based approaches to identify potential protein targets associated with diseases.

## 🎯 Features

- Multiple prediction algorithms:
  - Random Walk
  - Neighborhood-based approach
  - DIAMOnD algorithm
  - Node2Vec embedding
- Comprehensive evaluation metrics
- Community structure analysis
- Interactive visualizations
- Performance comparison

## 📊 Implemented Metrics

- Recall@k (k=10,20,50,100)
- Mean Reciprocal Rank (MRR)
- Average Precision (AP)
- Community clustering
- Disease protein connectivity

## 🚀 Getting Started

### Prerequisites

```bash
python -m pip install -r requirements.txt
```

Required packages:
- pandas
- networkx
- numpy
- scipy
- scikit-learn
- matplotlib
- tqdm
- gensim

### Project Structure

```
├── Dataset/
│   ├── bio-pathways-network/
│   │   └── bio-pathways-network.csv
│   └── bio-pathways-associations/
│       └── bio-pathways-associations.csv
├── pathways/
│   ├── prediction/
│   │   ├── randomWalk.py
│   │   ├── neighborhood.py
│   │   ├── diamond.py
│   │   └── node2vec_predict.py
│   ├── characterization/
│   │   └── calculate_community_scores.py
│   ├── run_predictions.py
│   └── plot_metrics.py
├── results/
└── README.md
```

### Running the Analysis

1. Clone the repository:
```bash
git clone https://github.com/atishay08/SIN_Project.git
cd disease-protein-prediction
```

2. Run the main prediction script:
```bash
python pathways/run_predictions.py
```

3. Generate visualization plots:
```bash
python pathways/plot_metrics.py
```

## 📈 Results

The project evaluates different algorithms on Liver Carcinoma dataset:

### Performance Metrics
- Recall@k measurements
- MRR (Mean Reciprocal Rank)
- AP (Average Precision)

### Community Structure
- Number of disease proteins
- Disease clustering coefficient
- Average disease degree
- Largest component size

## 📊 Visualization

The project includes two types of visualizations:
1. Recall@k performance curves
2. MRR and AP comparison plots

## 📝 Output

Results are saved in two formats:
1. Console output with detailed metrics
2. Text file in the results directory containing:
   - Evaluation metrics
   - Top 10 predictions
   - Community properties

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Network data from bio-pathways database
- Disease associations from validated sources
- Implementation based on established algorithms in the field

## 📞 Contact

Name - Atishay Jain
Project Link: [https://github.com/atishay08/SIN_Project]
