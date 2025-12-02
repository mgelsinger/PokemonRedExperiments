## Windows guide

Made by discord user @Aisiktir

1.) Python, https://www.python.org/ftp/python/3.11.6/python-3.11.6-amd64.exe 
 
2.) Install it, make sure "Add Python to path" is enabled 

3.) Git, https://github.com/git-for-windows/git/releases/download/v2.42.0.windows.2/Git-2.42.0.2-64-bit.exe 
 
4.) Install it  

4.1)

![https://i.postimg.cc/DyJ1m4YG/image.png](https://i.postimg.cc/DyJ1m4YG/image.png)

4.2)

![https://i.postimg.cc/C5WBbJ6q/image.png](https://i.postimg.cc/C5WBbJ6q/image.png)

4.3)

![https://i.postimg.cc/hPd7KWfb/image.png](https://i.postimg.cc/hPd7KWfb/image.png)

4.4)

![https://i.postimg.cc/LsNJxmbR/image.png](https://i.postimg.cc/LsNJxmbR/image.png)

4.5)

![https://i.postimg.cc/1XN4Rw55/image.png](https://i.postimg.cc/1XN4Rw55/image.png)

4.6)

![https://i.postimg.cc/HjDnrnV0/image.png](https://i.postimg.cc/HjDnrnV0/image.png)

4.7)

![https://i.postimg.cc/W17mPLHY/image.png](https://i.postimg.cc/W17mPLHY/image.png)

4.8)

![https://i.postimg.cc/cJ9Qr7ZQ/image.png](https://i.postimg.cc/cJ9Qr7ZQ/image.png)

4.9)

![https://i.postimg.cc/638CR75F/image.png](https://i.postimg.cc/638CR75F/image.png)

4.10)

![https://i.postimg.cc/2jB4SBtZ/image.png](https://i.postimg.cc/2jB4SBtZ/image.png)

4.11)

![https://i.postimg.cc/sD4GMP4v/image.png](https://i.postimg.cc/sD4GMP4v/image.png)

4.12)

![https://i.postimg.cc/bvWs7mpF/image.png](https://i.postimg.cc/bvWs7mpF/image.png)


5.) Microsoft C++ Build Tools, https://visualstudio.microsoft.com/visual-cpp-build-tools/ 

6.) Install it
![https://i.postimg.cc/Yq15fqds/image.png](https://i.postimg.cc/Yq15fqds/image.png)
 
Windows 11
![https://i.postimg.cc/VkKHDSnD/image.png](https://i.postimg.cc/VkKHDSnD/image.png)
 
Windows 10
![https://i.postimg.cc/43KwPWJx/image.png](https://i.postimg.cc/43KwPWJx/image.png)

7.) Open cmd 
 
7.1) Move into the repository root, depending where you've downloaded it, something like this: 
  
```cd "C:\Users\YourWindowsUsername\Downloads\PokemonRedExperiments"```
 
7.2) Install Python packages (CUDA/NVIDIA assumed):  
```pip install -r requirements.txt```
 
7.3) Run training (streaming on by default):  
```python training\train_ppo.py --rom PokemonRed.gb --state init.state --run-name my_run```

7.4) Run a trained checkpoint (loads latest zip in runs/ if not provided):  
```python training\play_checkpoint.py --rom PokemonRed.gb --state init.state```

7.5) Smoke test (quick sanity check):  
```python tools\smoke_test.py --rom PokemonRed.gb --state init.state```

7.6) Simple dashboard (view runs and compare commands):  
```python tools\serve_dashboard.py --port 8000```

If you already had Python installed and the `python` command resolves to another version, you can call it directly:

```
"%localappdata%\Programs\Python\Python311\Scripts\pip.exe" install -r requirements.txt
"%localappdata%\Programs\Python\Python311\python.exe" training\train_ppo.py --rom PokemonRed.gb --state init.state --run-name my_run
"%localappdata%\Programs\Python\Python311\python.exe" training\play_checkpoint.py --rom PokemonRed.gb --state init.state
"%localappdata%\Programs\Python\Python311\python.exe" tools\smoke_test.py --rom PokemonRed.gb --state init.state
"%localappdata%\Programs\Python\Python311\python.exe" tools\serve_dashboard.py --port 8000
```
