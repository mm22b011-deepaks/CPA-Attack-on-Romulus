# Correlation Power Analysis (CPA) on Romulus-N  
![CPA Logo](https://github.com/mm22b011-deepaks)  

### Authors  
**Deepak S (MM22B011)**  
**Monish M (CS23M003)**  

Contact: <deepaksridhar13@gmail.com>  
GitHub: [Mini Militia GitHub Repository](https://github.com/mm22b011-deepaks)

---

## Project Overview  
This repository implements a Correlation Power Analysis (CPA) attack on the **Romulus-N encryption algorithm** using Python. It demonstrates how power traces and plaintext data can be utilized to deduce the encryption key, focusing on the first two rounds of the SKINNY block cipher. The analysis is performed using `.npy` files containing the plaintext and power trace data.

---

## Features  
1. **First Round Key Recovery (K0 to K7):**  
   Leverages CPA techniques to extract key bytes from the ART (AddRoundTweakey) operation in the first round.  

2. **Second Round Key Recovery (K8 to K15):**  
   Extends the attack to the second round using advanced CPA techniques and Tweak Key computations.  

---
## Prerequisites  
- **Python 3.8+**  
- Required Python packages:  
  ```bash
  pip install numpy
  ```  
- `.npy` files for plaintext and power traces:  
  - Download from [this link](https://drive.google.com/file/d/1OvWAiAxAIXmww4Eou_vutsxElzdf0cPV/view?usp=sharing).  

---

## Running the Project  
1. Clone the repository:  
   ```bash
   git clone https://github.com/mm22b011-deepaks/cpa-romulus-n.git
   cd cpa-romulus-n
   ```  

2. Ensure the required `.npy` files are in the same directory as the script.  

3. Run the CPA attack script:  
   ```bash
   python Solution.py
   ```  

4. The deduced encryption key will be printed to the terminal.  

