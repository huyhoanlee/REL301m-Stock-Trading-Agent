# Stock trading using Reinforcement Learning

## Nội dung

- [Giới thiệu](#giới-thiệu)
- [Các yêu cầu hệ thống](#các-yêu-cầu-hệ-thống)
- [Hướng dẫn cài đặt](#hướng-dẫn-cài-đặt)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)

## Giới thiệu

Dự án này nhằm mục đích ứng dụng Reinforcement Learning trong bài toán mua bán chứng khoán.

## Các yêu cầu hệ thống

- Các thư viện cần thiết trong `requirements.txt`

## Hướng dẫn cài đặt

1. **Clone repo**:
   ```bash
   git clone https://github.com/huyhoanlee/REL301m-Stock-Trading-Agent.git
   ```

2. **Install các thư viện cần thiết trong file requirements**:
   ```bash
   pip install requirements.txt
    ```

## Hướng dẫn sử dụng

Sử dụng data `gmedata.csv` làm toy datasets hoặc đổi config path custom data.

1. **Train agent bằng phương pháp values-based**:
   ```bash
   python train_agent_values_based.py
    ```
2. **Train agent bằng phương pháp Polocy gradients**:
   ```bash
   python train_agent_DPG.py
   ```
3. **Inference model (use code correspond with trained agent, or can use notebook)**:
    ```bash
   python inference_DPG.py
   or 
   python inference_values-based.py
   ```