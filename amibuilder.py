import os
import sys
import logging
import logutils
import logutils.colorize
import shlex
import subprocess
import tempfile
import stat
import boto
import boto.utils
import boto.s3.connection
import boto.ec2.connection
from boto.ec2.blockdevicemapping import (
    EBSBlockDeviceType, BlockDeviceMapping)
import time
import select
import glob
import zc.thread
import docker
import functools
import urlparse
import re
import SocketServer
import SimpleHTTPServer
import tarfile
import multiprocessing.pool
import string
import random
import cliff.app
import cliff.command
from cliff.command import Command
import cliff.commandmanager
import shutil


try:
    from hashlib import md5
except:
    from md5 import md5


logutils.colorize.ColorizingStreamHandler.level_map = {
    logging.DEBUG: (None, 'cyan', False),
    logging.INFO: (None, 'green', False),
    logging.WARNING: (None, 'yellow', False),
    logging.ERROR: (None, 'magenta', False),
    logging.CRITICAL: (None, 'red', False)
    }

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
color_handler = logutils.colorize.ColorizingStreamHandler()
color_handler.setLevel(logging.DEBUG)
fmt = logging.Formatter('%(asctime)-15s %(message)s')
color_handler.setFormatter(fmt)
log.addHandler(color_handler)


dockerc = docker.Client(
    'unix://var/run/docker.sock', version='1.6', timeout=300)


def return_to_origin(fn):
    def decorator(*xs, **kw):
        origdir = os.getcwd()
        try:
            return fn(*xs, **kw)
        finally:
            if os.getcwd() != origdir:
                os.chdir(origdir)
    return decorator


def gotroot(fn):
    def decorator(*xs, **kw):
        if os.geteuid() != 0:
            raise AMIBuilderException('you should be root')
        return fn(*xs, **kw)


def split_file(filename, split_size=None, maxsplit=-1, suffix='', buf=2048):
    '''size in Bytes
    '''
    assert split_size is not None or maxsplit > 0
    size = os.path.getsize(filename)

    if split_size is None:
        split_size = size // maxsplit

    if maxsplit < 0:
        maxsplit = size // split_size

    cnt = 1
    with open(filename, 'rb') as fp:

        def new_name(cnt):
            return os.path.basename(filename) + '.' + suffix + str(cnt)

        while cnt <= maxsplit - 1:
            log.info('writing part no %s' % cnt)
            with open(new_name(cnt), 'wb') as partfp:
                for _ in range(split_size // buf):
                    partfp.write(fp.read(buf))
                else:
                    partfp.write(fp.read(split_size % buf))

            cnt += 1
        else:
            log.info('writing part no %s' % cnt)
            with open(new_name(cnt), 'wb') as partfp:
                data = fp.read(buf)
                while data:
                    partfp.write(data)
                    data = fp.read(buf)

        assert fp.read(1) == '', 'there should not be any byte left'


@gotroot
@return_to_origin
def make_ebs_based_image(ami_name, docker_image_name, fstype='ext3',
        mount_point=None, desc='', arch='x86_64', kernel=None, disk_size=10240):
    '''size of ebs is passed in Mb'''

    availability_zone = boto.utils.get_instance_metadata()[
        'placement']['availability-zone']
    instance_id = boto.utils.get_instance_metadata()['instance_id']

    # needs a ec2 connection here
    conn = boto.ec2.connection.EC2Connection()
    vol = conn.create_volume(disk_size, availability_zone)

    devpath = random.choice(
        [devp for devp in map(lambda x: '/dev/sd%s' % x, string.ascii_lowercase)
            if not os.path.exists(devp)]
        )

    vol.attach(instance_id, devpath)
    run('/sbin/mkfs -t %s %s' % (fstype, devpath))
    if mount_point is None:
        mount_point = tempfile.mkdtemp('ebs-based-mount-point')

    # copy files
    cid = dockerc.create_container(
        image=docker_image_name,
        command='/bin/bash',
        tty=True,
        volume=['dev']
        )

    export_fileobj = dockerc.export(cid)
    run('/bin/mount -t %s %s %s' % (fstype, devpath, mount_point))
    try:
        with tempfile.TemporaryFile() as fp:
            data = export_fileobj.read(2048)
            while data:
                fp.write(data)
                data = export_fileobj.read(2048)
            else:
                # if it goes well, seek to 0
                fp.seek(0)

        tar = tarfile.open(fileobj=fp)
        os.chdir(mount_point)
        tar.extractall()
        tar.close()
        os.chdir(os.pardir)
    finally:
        run('/bin/umount %s' % mount_point)

    vol.detach()
    snapshot = vol.create_snapshot('initial snapshot for ebs')
    ebs = EBSBlockDeviceType()
    ebs.snapshot_id = snapshot.id
    block_map = BlockDeviceMapping()
    block_map['/dev/sda1'] = ebs
    ami = conn.register_image(
        ami_name,
        description=desc,
        architecture=arch,
        kernel_id=kernel,
        root_device_name='dev/sda1',
        block_device_map=block_map
        )

    log.info('ebs-based ami: %s' % ami.id)


def retry(exceptions=None, max_retries=5, initial_delay=1, backoff_factor=2):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*xs, **kw):
            cnt, retry_delay = max_retries, initial_delay
            while cnt > 1:
                try:
                    return fn(*xs, **kw)
                except exceptions or Exception, ex:
                    log.warn('%s, retrying in %d sec' % (str(ex), retry_delay))
                time.sleep(retry_delay)
                retry_delay *= backoff_factor
                cnt -= 1
            # last try
            return fn(*xs, **kw)
        return wrapper
    return decorator


class AMIBuilderException(Exception):
    '''error occurred'''


def md5sum(filename, block_size=65536):
    assert os.path.exists(filename)

    hash_ = md5()
    with open(filename, 'r') as fp:
        buf = fp.read(block_size)
        while len(buf) > 0:
            hash_.update(buf)
            buf = fp.read(block_size)

    return hash_.hexdigest()


def serve(webdir):
    '''simple httpd server that servers
    out of a directory
    '''
    os.chdir(webdir)

    httpd = SocketServer.TCPServer(
        ('127.0.0.1', 0),
        SimpleHTTPServer.SimpleHTTPRequestHandler
        )

    port = httpd.socket.getsockname()[1]
    zc.thread.Thread(httpd.serve_forever)
    return port, httpd


@retry(boto.exception.S3ResponseError)
def download_file(bucket_name, keyname, filename=None):
    conn = boto.s3.connection.S3Connection()
    bucket = conn.get_bucket(bucket_name)
    keyob = bucket.get_key(keyname)

    if keyob is None:
        raise IOError('key does not exist')

    if not filename:
        filename = os.path.basename(keyname)

    # this try/except should go in a retry decorator
    keyob.get_contents_to_filename(filename)

    if md5sum(filename) != keyob.etag[1:-1]:
        raise AMIBuilderException('md5 does not match')


@retry(boto.exception.S3ResponseError)
def upload_file(filename, bucket_name, keyname):
    assert os.path.exists(filename)

    conn = boto.s3.connection.S3Connection()
    bucket = conn.get_bucket(bucket_name)
    keyob = bucket.get_key(keyname)

    if keyob is not None:
        if md5sum(filename) == keyob.etag[1:-1]:
            log.info('already in s3')
            return

    keyob = boto.s3.key.Key(bucket)
    keyob.key = keyname

    # a retry decorator should catch any exception
    # raised by this
    keyob.set_contents_from_filename(filename, policy='public-read')


@return_to_origin
def upload_bundle(manifest_path, bucket_name):
    '''upload bundle using a threadpool
    '''
    assert os.path.isfile(manifest_path)
    image_name = os.path.basename(manifest_path)[0:-13]
    topdir = os.path.dirname(manifest_path)

    os.chdir(topdir)

    to_upload = glob.glob(image_name + '*')
    pool = multiprocessing.pool.ThreadPool(30)

    def upload_file_helper(filename):
        return upload_file(
            filename,
            bucket_name='zcamis',
            keyname=os.path.join('bundles', image_name, filename)
            )

    rv = pool.map(upload_file_helper, to_upload)

    log.info('done uploading %s', str(rv))


def run(command_line, timeout=sys.maxint, fp=None):
    argslist = shlex.split(command_line)
    rpipe, wpipe = os.pipe()
    start_time = time.time()

    ps = subprocess.Popen(
        argslist, stdout=wpipe, stderr=wpipe, shell=False
        )

    while timeout > time.time() - start_time:
        rfds, wfds, xfds = select.select([rpipe], [], [], 0.1)
        for fd in rfds:
            data = os.read(fd, 10000)
            if fp is not None:
                fp.write(data)
            else:
                sys.stdout.write(data)
                sys.stdout.flush()

        if ps.poll() is not None:
            os.close(rpipe)
            os.close(wpipe)
            break

    if ps.returncode != 0:
        raise AMIBuilderException(
            'Run(cmd=%s, retcode=%s)' % (command_line, ps.returncode)
            )


def make_sparse(filename, size=None):
    '''gets the path of file and the size in MB
    raise an exeception if file exists
    '''
    assert isinstance(size, int), 'needs an integer'

    if os.path.exists(filename):
        raise IOError('cant make sparse file, one exists already')

    with open(filename, 'wb') as fp:
        fp.seek(size * 1024 * 1024 - 1)
        fp.write('\0')


@gotroot
@return_to_origin
def make_file_image_from_docker_image(filename, size=None,
        docker_image_name=None, fstype='ext3', mount_point=None):

    cid = dockerc.create_container(
        image=docker_image_name,
        command='/bin/bash',
        tty=True,
        volumes=['/dev']
        )['Id']

    if mount_point is not None:
        if os.path.exists(mount_point):
            if not os.path.isdir(mount_point):
                raise AMIBuilderException(
                    'mount point exists and is not a dir')
        else:
            os.makedirs(mount_point)
    else:
        mount_point = tempfile.mkdtemp(prefix='ec2-inst-mount-point')

    make_sparse(filename, size)
    run('/sbin/mkfs -t %s -F %s' % (fstype, filename))

    run('/bin/mount -o loop %s %s' % (filename, mount_point))
    export_fileobj = dockerc.export(cid)
    try:
        with tempfile.TemporaryFile() as fp:
            data = export_fileobj.read(2048)
            while data:
                fp.write(data)
                data = export_fileobj.read(2048)
            else:
                fp.seek(0)

            tar = tarfile.open(fileobj=fp)
            os.chdir(mount_point)
            tar.extractall()
            tar.close()
            # get out of mount_point
            os.chdir(os.pardir)
    finally:
        run('/bin/umount %s' % filename)


@gotroot
@return_to_origin
def make_devices(devdir):
    os.chdir(devdir)

    os.mknod('null', 0666 | stat.S_IFCHR, os.makedev(1, 3))
    os.mknod('zero', 0666 | stat.S_IFCHR, os.makedev(1, 5))
    os.mknod('random', 0666 | stat.S_IFCHR, os.makedev(1, 8))
    os.mknod('urandom', 0666 | stat.S_IFCHR, os.makedev(1, 9))
    os.mknod('tty', 0666 | stat.S_IFCHR, os.makedev(5, 0))
    os.mknod('tty0', 0666 | stat.S_IFCHR, os.makedev(4, 0))
    os.mknod('console', 0600 | stat.S_IFCHR, os.makedev(5, 1))
    os.mknod('full', 0666 | stat.S_IFCHR, os.makedev(1, 7))
    os.mknod('ptmx', 0666 | stat.S_IFCHR, os.makedev(5, 2))
    os.mkdir('pts', 0755)
    os.mkdir('shm', 0777)


@return_to_origin
def import_image(uri, image_name):
    '''Get archive from s3 or from filesystem or
    web and insert it into docker. This is a clean
    image
    '''
    uri = urlparse.urlparse(uri)
    tempdir = tempfile.mkdtemp(prefix='import')
    os.chdir(tempdir)

    if uri.scheme:
        if uri.scheme.lower() == 's3':
            download_file(
                bucket_name=uri.netloc,
                keyname=uri.path,
                )

        if uri.scheme.lower().startswith('http'):
            # download resource available
            pass

    else:
        # means that the file is in the localhost
        shutil.copy(uri.path, tempdir)

    filename = os.path.basename(uri.path)
    dockerc.import_image(filename, repository=image_name)


def create(image_name, script_path, hostname=None, new_image_name=None):
    '''creates a container base on image_name
    inserts script_path into it then runs it
    returns the created image id
    '''
    # httpd.server_forever and shutdown can go
    # into a ctx mgr
    port, httpd = serve(
        os.path.dirname(os.path.abspath(script_path))
        )

    script_name = os.path.basename(script_path)
    resp = dockerc.insert(
        image_name,
        'http://127.0.0.1:%d/%s' % (port, script_name),
        '/'+script_name
        )

    httpd.shutdown()

    # parsing the response with a re instead of json.loads
    # to avoid errors
    intermediate_image = re.search(
        '{\s*"[Ii]d"\s*:\s*"(.*)"\s*}', resp).group(1)

    devdir = tempfile.mkdtemp(prefix='docker-devices')

    make_devices(devdir)

    cid = dockerc.create_container(
        image=intermediate_image,
        command='/bin/bash /%s' % script_name,
        hostname=hostname,
        tty=True,
        volumes=['/dev']
        )[u'Id']

    dockerc.start(cid, binds={devdir: '/dev'})

    # this should be probably done with any and map
    while True:
        time.sleep(1)
        data = dockerc.containers()
        if not data:
            break
        for elem in data:
            if elem[u'Id'].startswith(cid):
                break
        else:
            break

    resp = dockerc.commit(cid, repository=new_image_name)
    log.info('new image: %s-> %s' % (new_image_name, resp['Id']))


class AmiBuilder(cliff.app.App):

    def __init__(self):
        super(AmiBuilder, self).__init__(
            description='ami builder',
            version='0.1',
            command_manager=cliff.commandmanager.CommandManager(
                'cliff.amibuilder')
            )

    def initialize_app(self, args):
        log.debug('initialize app')


class Bundle(Command):
    '''bundle a disk image, upload it to s3
    and register it'''

    def get_parser(self, program_name):
        parser = super(Command, self).get_parser(program_name)
        parser.add_argument('--ami-name', required=True)
        parser.add_argument('--image-path', required=True)
        parser.add_argument('--arch', default='x86_64')
        parser.add_argument('--prefix')
        parser.add_argument('--bucket-name')
        parser.add_argument('--kernel')
        parser.add_argument('--desc')
        parser.add_argument('--region')
        return parser

    def take_action(self, options):
        todir = tempfile.mkdtemp(prefix='bundle-image')
        log.info('bundling image %s', todir)

        run(' '.join([
            '/usr/local/bin/ec2-bundle-image',
            '--cert %s' % boto.config.get('Credentials', 'cert'),
            '--privatekey %s' % boto.config.get('Credentials', 'pkey'),
            '--user %s' % boto.config.get('Credentials', 'user'),
            '--image %s' % options.image_path,
            '--destination %s' % todir,
            '--arch %s' % options.arch,
            '--prefix %s' % options.prefix,
            ]))

        upload_bundle(
            os.path.join(todir, options.prefix + '.manifest.xml'),
            bucket_name=options.bucket_name
            )

        conn = boto.ec2.connection.EC2Connection()

        amiid = conn.register_image(
            name=options.ami_name,
            description=options.desc,
            image_location=os.path.join(
                options.bucket_name,
                'bundles',
                options.prefix,
                options.prefix + '.manifest.xml'
                ),
            architecture=options.arch,
            kernel_id=options.kernel,
            )

        imglist = conn.get_all_images([amiid])
        conn.create_tags(imglist, dict(Name=options.ami_name))

        log.info('ami id: %s' % amiid)


# import into docker
class Import(cliff.command.Command):
    '''import a tar.gz file in s3 into docker'''

    def get_parser(self, program_name):
        parser = super(Command, self).get_parser(program_name)
        # s3 uri
        parser.add_argument('-u', '--uri', required=True)
        parser.add_argument('-n', '--docker-image-name')
        return parser

    def take_action(self, options):
        imgid = import_image(options.uri, options.image_name)
        log.info('ami id:', imgid)


class Upload(cliff.command.Command):
    '''upload a file to s3'''
    def get_parser(self, program_name):
        parser = super(Command, self).get_parser(program_name)
        parser.add_argument('-f', '--filename', required=True,
                            help='file to upload')
        parser.add_argument('-n', '--new-name', help='new name in s3')
        return parser

    def take_action(self, options):
        log.info('uploading file', options.filename)
        curr_name = os.path.basename(options.file_to_upload)

        upload_file(
            filename=options.file_to_upload,
            bucket_name='zcamis',
            keyname='clean_images/%s' % (options.new_name or curr_name)
            )


class Run(Command):
    '''run script in docker image and create a new
    docker image from the resulting container'''
    def get_parser(self, program_name):
        parser = super(Command, self).get_parser(program_name)
        parser.add_argument('-n', '--new-image-name', required=True)
        parser.add_argument('-c', '--image-name', required=True)
        parser.add_argument('-s', '--script-path', required=True)
        parser.add_argument('-h', '--hostname')
        return parser

    def take_action(self, options):
        log.info('create new docker image from script')

        create(
            options.image_name,
            options.script_path,
            options.hostname,
            options.new_image_name
            )


class MakeFs(cliff.command.Command):
    '''make a fs in a disk image from a docker image'''
    def get_parser(self, program_name):
        parser = super(Command, self).get_parser(program_name)
        parser.add_argument('-f', '--filename', required=True)
        parser.add_argument('-s', '--size', required=True)
        parser.add_argument('-i', '--docker-image', required=True)
        parser.add_argument('-t', '--fstype')
        parser.add_argument('-m', '--mount-point')
        return parser

    def take_action(self, options):
        make_file_image_from_docker_image(
            options.filename,
            options.size,
            options.docker_image,
            options.fstype,
            options.mount_point
            )


class Extract(Command):
    '''extract tar.gz file into a disk image'''
    def get_parser(self, program_name):
        parser = super(Command, self).get_parser(program_name)
        parser.add_argument('-i', '--disk-image-path', required=True)
        parser.add_argument('-f', '--file-to-extract', required=True)
        parser.add_argument('-t', '--fstype', default='ext3', required=True)
        return parser

    @return_to_origin
    def take_action(self, options):
        mount_point = tempfile.mkdtemp(prefix='extract-files-from-image')
        file_to_extract = os.path.abspath(options.file_to_extract)
        assert not os.path.exists(file_to_extract)

        run('/bin/mount -o loop -t %s %s %s' % (options.fstype,
            options.disk_image_path, mount_point))
        os.chdir(mount_point)
        try:
            tgz = tarfile.open(file_to_extract, 'w:gz')
            for fname in os.listdir('.'):
                log.info('archiving...')
                tgz.add(fname)
            tgz.close()
        finally:
            os.chdir(os.pardir)
            run('/bin/umount %s' % options.disk_image_path)


class SplitFile(Command):
    '''split file in parts'''

    def get_parser(self, program_name):
        parser = super(SplitFile, self).get_parser(program_name)
        parser.add_argument('--split-size', type=int, help='split size')
        parser.add_argument('--maxsplit', type=int, help='max split')
        parser.add_argument('--suffix', help='suffix')
        parser.add_argument('filename', nargs='?', help='suffix')
        return parser

    def take_action(self, options):
        log.info('splitting %s' % options.filename)
        split_file(
            options.filename,
            options.split_size,
            options.maxsplit,
            options.suffix
            )


def main(args=None):
    app = AmiBuilder()
    return app.run(args or sys.argv[1:])


if __name__ == '__main__':
    sys.exit(main())
