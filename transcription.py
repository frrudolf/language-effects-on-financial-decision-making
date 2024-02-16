# This code comes directly from Kevin Stratvert (January 18, 2023) and is altered to my needs
# install Whisper AI
!pip install git+https://github.com/openai/whisper.git
!sudo apt update && sudo apt install ffmpeg
# Collecting git+https://github.com/openai/whisper.git
# Cloning https://github.com/openai/whisper.git to /tmp/pip-req-build-_r9nxzsi
# Running command git clone --filter=blob:none --quiet https://github.com/openai/whisper.git /tmp/pip-req-build-_r9nxzsi
# Resolved https://github.com/openai/whisper.git to commit e8622f9afc4eba139bf796c210f5c01081000472
# Installing build dependencies ... done
# Getting requirements to build wheel ... done
# Preparing metadata (pyproject.toml) ... done
# Requirement already satisfied: triton==2.0.0 in /usr/local/lib/python3.10/dist-packages (from openai-whisper==20230314) (2.0.0)
# Requirement already satisfied: numba in /usr/local/lib/python3.10/dist-packages (from openai-whisper==20230314) (0.56.4)
# Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from openai-whisper==20230314) (1.23.5)
# Requirement already satisfied: torch in /usr/local/lib/python3.10/dist-packages (from openai-whisper==20230314) (2.0.1+cu118)
# Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from openai-whisper==20230314) (4.66.1)
# Requirement already satisfied: more-itertools in /usr/local/lib/python3.10/dist-packages (from openai-whisper==20230314) (10.1.0)
# Collecting tiktoken==0.3.3 (from openai-whisper==20230314)
# Downloading tiktoken-0.3.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.7 MB)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.7/1.7 MB 20.3 MB/s eta 0:00:00
# Requirement already satisfied: regex>=2022.1.18 in /usr/local/lib/python3.10/dist-packages (from tiktoken==0.3.3->openai-whisper==20230314) (2023.6.3)
# Requirement already satisfied: requests>=2.26.0 in /usr/local/lib/python3.10/dist-packages (from tiktoken==0.3.3->openai-whisper==20230314) (2.31.0)
# Requirement already satisfied: cmake in /usr/local/lib/python3.10/dist-packages (from triton==2.0.0->openai-whisper==20230314) (3.27.4.1)
# Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from triton==2.0.0->openai-whisper==20230314) (3.12.2)
# Requirement already satisfied: lit in /usr/local/lib/python3.10/dist-packages (from triton==2.0.0->openai-whisper==20230314) (16.0.6)
# Requirement already satisfied: llvmlite<0.40,>=0.39.0dev0 in /usr/local/lib/python3.10/dist-packages (from numba->openai-whisper==20230314) (0.39.1)
# Requirement already satisfied: setuptools in /usr/local/lib/python3.10/dist-packages (from numba->openai-whisper==20230314) (67.7.2)
# Requirement already satisfied: typing-extensions in /usr/local/lib/python3.10/dist-packages (from torch->openai-whisper==20230314) (4.5.0)
# Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch->openai-whisper==20230314) (1.12)
# Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch->openai-whisper==20230314) (3.1)
# Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch->openai-whisper==20230314) (3.1.2)
# Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests>=2.26.0->tiktoken==0.3.3->openai-whisper==20230314) (3.2.0)
# Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.26.0->tiktoken==0.3.3->openai-whisper==20230314) (3.4)
# Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.26.0->tiktoken==0.3.3->openai-whisper==20230314) (2.0.4)
# Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.26.0->tiktoken==0.3.3->openai-whisper==20230314) (2023.7.22)
# Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch->openai-whisper==20230314) (2.1.3)
# Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->torch->openai-whisper==20230314) (1.3.0)
# Building wheels for collected packages: openai-whisper
# Building wheel for openai-whisper (pyproject.toml) ... done
# Created wheel for openai-whisper: filename=openai_whisper-20230314-py3-none-any.whl size=798395 sha256=4c0f50fcc95bb0bdee12c78ea416b8a329499e92d7f33b7bc9319e5890d82ef5
# Stored in directory: /tmp/pip-ephem-wheel-cache-25jlk0x7/wheels/8b/6c/d0/622666868c179f156cf595c8b6f06f88bc5d80c4b31dccaa03
# Successfully built openai-whisper
# Installing collected packages: tiktoken, openai-whisper
# Successfully installed openai-whisper-20230314 tiktoken-0.3.3
# Get:1 https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/ InRelease [3,626 B]
# Hit:2 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64  InRelease
# Get:3 http://security.ubuntu.com/ubuntu jammy-security InRelease [110 kB]
# Hit:4 http://archive.ubuntu.com/ubuntu jammy InRelease
# Get:5 http://archive.ubuntu.com/ubuntu jammy-updates InRelease [119 kB]
# Get:6 https://ppa.launchpadcontent.net/c2d4u.team/c2d4u4.0+/ubuntu jammy InRelease [18.1 kB]
# Get:7 http://security.ubuntu.com/ubuntu jammy-security/main amd64 Packages [962 kB]
# Get:8 http://archive.ubuntu.com/ubuntu jammy-backports InRelease [109 kB]
# Hit:9 https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy InRelease
# Get:10 http://security.ubuntu.com/ubuntu jammy-security/restricted amd64 Packages [1,059 kB]
# Get:11 http://archive.ubuntu.com/ubuntu jammy-updates/restricted amd64 Packages [1,079 kB]
# Hit:12 https://ppa.launchpadcontent.net/graphics-drivers/ppa/ubuntu jammy InRelease
# Get:13 http://archive.ubuntu.com/ubuntu jammy-updates/universe amd64 Packages [1,254 kB]
# Get:14 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 Packages [1,230 kB]
# Hit:15 https://ppa.launchpadcontent.net/ubuntugis/ppa/ubuntu jammy InRelease
# Get:16 http://archive.ubuntu.com/ubuntu jammy-backports/universe amd64 Packages [28.1 kB]
# Get:17 https://ppa.launchpadcontent.net/c2d4u.team/c2d4u4.0+/ubuntu jammy/main Sources [2,180 kB]
# Get:18 https://ppa.launchpadcontent.net/c2d4u.team/c2d4u4.0+/ubuntu jammy/main amd64 Packages [1,119 kB]
# Fetched 9,271 kB in 7s (1,376 kB/s)
# Reading package lists... Done
# Building dependency tree... Done
# Reading state information... Done
# 16 packages can be upgraded. Run 'apt list --upgradable' to see them.
# Reading package lists... Done
# Building dependency tree... Done
# Reading state information... Done
# ffmpeg is already the newest version (7:4.4.2-0ubuntu0.22.04.1).
# 0 upgraded, 0 newly installed, 0 to remove and 16 not upgraded.


# After having downloaded and renamed all selected TV spots, I could use Whisper AI to transcribe each one individually

!whisper "Bildschirmaufnahme42.mov" --model medium.en --output_format txt
# [00:00.000 --> 00:04.440]  with the savings rates offered at some banks your money almost seems like it's
# [00:04.440 --> 00:10.320]  frozen but not here with Capital One you can open a new savings account in about
# [00:10.320 --> 00:14.760]  five minutes and earn five times the national average open one from here or
# [00:14.760 --> 00:19.520]  here in a Capital One cafe plus there are no fees or minimums on savings or
# [00:19.520 --> 00:25.200]  checking accounts because that's how it should be this is banking reimagined
# [00:25.200 --> 00:28.600]  what's in your wallet
