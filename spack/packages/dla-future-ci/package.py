# Copyright 2013-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from os import path, environ
from spack import *

try:
	server_url = environ['GITHUB_SERVER_URL']
	repo = environ['GITHUB_REPOSITORY']
        commit_sha  = environ['GITHUB_SHA']
	git_url = path.join(server_url, repo)
except KeyError as e:
       raise InstallError('This package can only be installed within a GitHub action run. {} environment variable must be defined.'.format(e))


class DlaFutureCi(CMakePackage):
    """The DLAF package provides DLA-Future library: Distributed Linear Algebra with Future"""

    homepage = "https://github.com/eth-cscs/DLA-Future.git/wiki"
    git      = git_url

    maintainers = ['teonnik', 'Sely85']

    sources_root = path.abspath(path.join(path.dirname(__file__), '../../../'))
    version('current-ci', commit=commit_sha)

    variant('cuda', default=False,
            description='Use the GPU/cuBLAS back end.')
    variant('doc', default=False,
            description='Build documentation.')

    depends_on('mpi')
    depends_on('blaspp')
    depends_on('lapackpp')
    depends_on('hpx@1.4.0:1.4.1 cxxstd=14 networking=none')
    depends_on('cuda', when='+cuda')

    def cmake_args(self):
       spec = self.spec

       if '^mkl' in spec:
           args = ['-DDLAF_WITH_MKL=ON']
       else:
           args = ['-DDLAF_WITH_MKL=OFF']
           args.append('-DLAPACK_TYPE=Custom')
           args.append('-DLAPACK_LIBRARY={} {}'.format(spec['lapack'].libs.ld_flags, spec['blas'].libs.ld_flags))

       if '+cuda' in spec:
           args.append('-DDLAF_WITH_CUDA=ON')

       if self.run_tests:
           args.append('-DDLAF_WITH_TEST=ON')
       else:
           args.append('-DDLAF_WITH_TEST=OFF')

       if '+doc' in spec:
           args.append('-DBUILD_DOC=on')

       return args
