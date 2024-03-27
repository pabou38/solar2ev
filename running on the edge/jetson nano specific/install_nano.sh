echo "Jetson Nano post installation"

sudo apt-get install nano
sudo apt install dos2unix
sudo apt install tmux
sudo apt install htop

sudo apt install python3-pip
pip3 install --upgrade pip

pip3 install virtualenv

echo 'configure networking'

# cannot be eth0, is a connection name
sudo nmcli con mod "Wired connection 1" ipv4.addresses 192.168.1.242/24
sudo nmcli con mod Freebox-382B6B   ipv4.addresses 192.168.1.243/24

# gateway, DNS
sudo nmcli con mod "Wired connection 1" ipv4.gateway 192.168.1.1
sudo nmcli con mod "Wired connection 1" ipv4.dns 192.168.1.1
sudo nmcli con mod "Wired connection 1" ipv4.method manual


sudo nmcli con mod Freebox-382B6B  ipv4.gateway 192.168.1.1
sudo nmcli con mod Freebox-382B6B  ipv4.dns 192.168.1.1
sudo nmcli con mod Freebox-382B6B  ipv4.method manual

echo 'tensorflow'

# get it from nvidia, standard aarch64 wheel may not have GPU enabled

# https://forums.developer.nvidia.com/t/official-tensorflow-for-jetson-nano/71770

pip install numpy==1.19.4    # otherwize core dump

# no venv, otherwize seem to fails h5py build
# no sudo
# pip3 install --verbose tensorflow-2.7.0+nv22.1-cp36-cp36m-linux_aarch64.whl 
