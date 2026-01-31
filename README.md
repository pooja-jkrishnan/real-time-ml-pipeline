# Real-Time Machine Learning Pipeline

This repository contains the implementation of a real-time machine learning pipeline developed as part of an MSc thesis.  
The project focuses on designing and evaluating an integrated system for streaming data ingestion, real-time feature computation, online inference, and continuous performance evaluation.

The goal of the study is to empirically examine how machine learning systems behave under real-time execution conditions, with particular emphasis on latency, throughput, and system-level integration challenges.

---

## Project Overview

The implemented pipeline supports:

- Continuous streaming data ingestion
- Stateful, window-based feature computation
- Real-time machine learning inference
- System-level metric collection and evaluation
- Reproducible experimentation using containerised execution

The system is designed as a modular, event-driven pipeline and evaluated using controlled real-time data streams.

---

## Architecture and Components

The main components of the system are:

- **Streaming Ingestion**  
  Apache Kafka is used to ingest and buffer real-time event streams.

- **Stream Processing**  
  Apache Spark Structured Streaming performs feature extraction and window-based aggregations.

- **Model Inference**  
  A Python-based service applies a pre-trained machine learning model to streaming features in real time.

- **Experiment Tracking**  
  MLflow is used to log run-level metadata, model information, and evaluation metrics.

- **Deployment Environment**  
  Docker is used to containerise system components and ensure consistent execution environments.

---

## Repository Structure

