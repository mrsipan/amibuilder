Build an AMI
============

1. Import an image into docker::

    sudo BOTO_CONFIG=/home/ben/.boto time bin/amibuilder -m -u \
    s3://zcamis/clean_images/centos-6.5-x86_64.tgz  -t \
    centos-6.5-x86_64-clean 

2. Build docker image taking another image as base and running a script::

    sudo BOTO_CONFIG=/home/ben/.boto time bin/amibuilder -r \
    --image-name \
    centos-6.5-x86_64-clean --new-image-name testc6image \
    -e hola -s ../ami-setup/centos-6-x86_64-setup.sh

3. Create an filesystem image to bundle::

    sudo BOTO_CONFIG=/home/ben/.boto time bin/amibuilder -q \
    -j hola.img -z 5000 -g testc6image

4. Bundle and upload::

    bin/amibuilder -i hola.img -a x86_64 -p testbenc65 -b zcamis \
        -k aki-88aa75e1

5. Upload a file to s3://zcamis/clean_images::

    bin/amibuilder -o -f centos6.tgz -n centos-6.5-x86_64.tz

6. Split an image file in 2::

    bin/amibuilder split --split-size=1048576 --maxsplit=2 \
        --suffix=f20 ../ksxen/f20.img



