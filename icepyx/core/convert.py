import os
import re
import h5py
import zarr
import pandas
import itertools
import numpy as np

class convert():
    np.seterr(invalid='ignore')
    def __init__(self, filename=None, directory='', reformat=None):
        self.filename = filename
        self.directory = os.path.expanduser(directory)
        self.reformat = reformat

    # PURPOSE: wrapper function for converting HDF5 files to another type
    def file_converter(self, **kwds):
        """
        Locally convert a HDF5 file to another format
        """
        if (self.reformat == 'zarr'):
            # output zarr file
            self.HDF5_to_zarr(**kwds)
        # elif (reformat == 'JPL'):
        #     # output JPL captoolkit formatted HDF5 files
        #     self.HDF5_to_JPL_HDF5(**kwds)
        elif self.reformat in ('csv','txt'):
            # output reduced files to ascii formats
            self.HDF5_to_ascii(**kwds)
        elif self.reformat in ('dataframe'):
            # output reduced files to pandas dataframe
            return self.HDF5_to_dataframe(**kwds)

    # PURPOSE: convert the HDF5 file to zarr copying all file data
    def HDF5_to_zarr(self, **kwds):
        """
        convert the HDF5 file to zarr copying all file data
        """
        # split extension from HDF5 file
        if isinstance(self.filename, str):
            fileBasename,fileExtension=os.path.splitext(self.filename)
        else:
            fileBasename,fileExtension=os.path.splitext(self.filename.filename)
        # output zarr file
        zarr_file = os.path.join(self.directory,'{0}.zarr'.format(fileBasename))
        # copy everything from the HDF5 file to the zarr file
        with h5py.File(self.filename,mode='r') as source:
            dest = zarr.open_group(zarr_file,mode='w')
            # value checks on output zarr
            if not hasattr(dest, 'create_dataset'):
                raise ValueError('dest must be a group, got {!r}'.format(dest))
            # for each key in the root of the hdf5 file structure
            for k in source.keys():
                self.copy_from_HDF5(source[k], dest, name=k)

    # PURPOSE: Copy a named variable from the HDF5 file to the zarr file
    def copy_from_HDF5(self, source, dest, name, **kwds):
        """Copy a named variable from the `source` HDF5 into the `dest` zarr"""
        if hasattr(source, 'shape'):
            # copy a dataset/array
            if dest is not None and name in dest:
                raise CopyError('an object {!r} already exists in destination '
                    '{!r}'.format(name, dest.name))
            # setup creation keyword arguments
            create_kwds = kwds.copy()
            # setup chunks option, preserve by default
            create_kwds.setdefault('chunks', source.chunks)
            # setup compression options
            # from h5py to zarr: use zarr default compression options
            create_kwds.setdefault('fill_value', source.fillvalue)
            # create new dataset in destination
            ds = dest.create_dataset(name, shape=source.shape,
                dtype=source.dtype, **create_kwds)
            # copy data going chunk by chunk to avoid loading in entirety
            shape = ds.shape
            chunks = ds.chunks
            chunk_offsets = [range(0, s, c) for s, c in zip(shape, chunks)]
            for offset in itertools.product(*chunk_offsets):
                sel = tuple(slice(o, min(s, o + c)) for o, s, c in
                    zip(offset, shape, chunks))
                ds[sel] = source[sel]
            # copy attributes
            attrs = {key:self.attributes_encoder(source.attrs[key]) for key in
                source.attrs.keys() if self.attributes_encoder(source.attrs[key])}
            ds.attrs.update(attrs)
        else:
            # copy a group
            if (dest is not None and name in dest and hasattr(dest[name], 'shape')):
                raise CopyError('an array {!r} already exists in destination '
                    '{!r}'.format(name, dest.name))
            # require group in destination
            grp = dest.require_group(name)
            # copy attributes
            attrs = {key:self.attributes_encoder(source.attrs[key]) for key in
                source.attrs.keys() if self.attributes_encoder(source.attrs[key])}
            grp.attrs.update(attrs)
            # recursively copy from source
            for k in source.keys():
                self.copy_from_HDF5(source[k], grp, name=k)

    # PURPOSE: reduce HDF5 files to beam groups and output to ascii
    def HDF5_to_ascii(self, **kwds):
        """
        convert the HDF5 file to beam-level ascii files copying reduced sets of data
        """
        # compile regular expression operator for extracting info from ICESat2 files
        rx = re.compile(r'(processed)?(ATL\d+)(-\d{{2}})?_(\d{4})(\d{2})(\d{2})'
            r'(\d{2})(\d{2})(\d{2})_(\d{4})(\d{2})(\d{2})_(\d{3})_(\d{2})(.*?).h5$')
        # split extension from HDF5 file
        # extract parameters from ICESat2 HDF5 file
        if isinstance(self.filename, str):
            fileBasename,fileExtension=os.path.splitext(self.filename)
            # extract parameters from ICESat2 HDF5 file
            SUB,PRD,HEM,YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,RL,VERS,AUX = \
                rx.findall(os.path.basename(self.filename)).pop()
        else:
            fileBasename,fileExtension=os.path.splitext(self.filename.filename)
            SUB,PRD,HEM,YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,RL,VERS,AUX = \
                rx.findall(os.path.basename(self.filename.filename)).pop()
        # output file suffix for csv or tab-delimited text
        delimiter = ',' if self.reformat == 'csv' else '\t'
        # copy bare minimum variables from the HDF5 file to the ascii file
        source = h5py.File(self.filename,mode='r')

        # find valid beam groups by testing for particular variables
        if (PRD == 'ATL06'):
            VARIABLE_PATH = ['land_ice_segments','segment_id']
        elif (PRD == 'ATL07'):
            VARIABLE_PATH = ['sea_ice_segments','height_segment_id']
        elif (PRD == 'ATL08'):
            VARIABLE_PATH = ['land_segments','segment_id_beg']
        elif (PRD == 'ATL10'):
            VARIABLE_PATH = ['freeboard_beam_segments','delta_time']
        elif (PRD == 'ATL12'):
            VARIABLE_PATH = ['ssh_segments']['delta_time']
        # create list of valid beams within the HDF5 file
        beams = []
        for gtx in [k for k in source.keys() if bool(re.match(r'gt\d[lr]',k))]:
            # check if subsetted beam contains data
            try:
                source['/'.join([gtx,*VARIABLE_PATH])]
            except KeyError:
                pass
            else:
                beams.append(gtx)

        # for each valid beam within the HDF5 file
        for gtx in sorted(beams):
            # create a column stack of valid output segment values
            if (PRD == 'ATL06'):
                var = source[gtx]['land_ice_segments']
                valid, = np.nonzero(var['atl06_quality_summary'][:] == 0)
                # variables for the output ascii file
                vnames = ['segment_id','delta_time','latitude','longitude',
                    'h_li','h_li_sigma']
                vformat = ('{1:0.0f}{0}{2:0.9f}{0}{3:0.9f}{0}{4:0.9f}{0}'
                    '{5:0.9f}{0}{6:0.9f}')
                # extract attributes for each variable
                vattrs = {}
                for i,v in enumerate(vnames):
                    vattrs[v] = {atn:atv for atn,atv in var[v].attrs.items()}
                    if (v == 'segment_id'):
                        vattrs[v]['precision'] = 'integer'
                        vattrs[v]['units'] = 'count'
                    else:
                        vattrs[v]['precision'] = 'double_precision'
                    vattrs[v]['comments'] = 'column {0:d}'.format(i+1)
                # column stack of valid output segment values
                output = np.column_stack([var[v][valid] for v in vnames])

            # output ascii file
            ascii_file = '{0}_{1}.{2}'.format(fileBasename,gtx,self.reformat)
            fid = open(os.path.join(self.directory,ascii_file),'w')
            # print YAML header to top of file
            fid.write('{0}:\n'.format('header'))
            # global attributes for file
            fid.write('  {0}:\n'.format('global_attributes'))
            for atn,atv in source.attrs.items():
                if atn not in ('Conventions','Processing Parameters','hdfversion',
                    'history','identifier_file_uuid'):
                    fid.write('    {0:22}: {1}\n'.format(atn,self.attributes_encoder(atv)))
            # beam attributes
            fid.write('\n  {0}:\n'.format('beam_attributes'))
            for atn,atv in source[gtx].attrs.items():
                if atn not in ('Description',):
                    fid.write('    {0:22}: {1}\n'.format(atn,self.attributes_encoder(atv)))
            # data dimensions
            fid.write('\n  {0}:\n'.format('dimensions'))
            nrow,ncol = np.shape(output)
            fid.write('    {0:22}: {1:d}\n'.format('segments',nrow))
            # non-standard attributes
            fid.write('\n  {0}:\n'.format('non-standard_attributes'))
            # value to convert to GPS seconds (seconds since 1980-01-06T00:00:00)
            fid.write('    {0:22}:\n'.format('atlas_sdp_gps_epoch'))
            atlas_sdp_gps_epoch = source['ancillary_data']['atlas_sdp_gps_epoch']
            for atn in ['units','long_name']:
                atv = self.attributes_encoder(atlas_sdp_gps_epoch.attrs[atn])
                fid.write('      {0:20}: {1}\n'.format(atn,atv))
            fid.write('      {0:20}: {1:0.0f}\n'.format('value',atlas_sdp_gps_epoch[0]))
            # print variable descriptions to YAML header
            fid.write('\n  {0}:\n'.format('variables'))
            for v in vnames:
                fid.write('    {0:22}:\n'.format(v))
                for atn in ['precision','units','long_name','comments']:
                    atv = self.attributes_encoder(vattrs[v][atn])
                    fid.write('      {0:20}: {1}\n'.format(atn,atv))
            # end of header
            fid.write('\n\n# End of YAML header\n')
            # print data to file
            for row in output:
                print(vformat.format(delimiter,*row),file=fid)
            # close the file
            fid.close()
        # close the source HDF5 file
        source.close()

    # PURPOSE: reduce HDF5 files to pandas dataframe
    def HDF5_to_dataframe(self, **kwds):
        """
        convert the HDF5 file to a pandas dataframe copying reduced sets of data
        """
        # compile regular expression operator for extracting info from ICESat2 files
        rx = re.compile(r'(processed)?(ATL\d+)(-\d{{2}})?_(\d{4})(\d{2})(\d{2})'
            r'(\d{2})(\d{2})(\d{2})_(\d{4})(\d{2})(\d{2})_(\d{3})_(\d{2})(.*?).h5$')
        # split extension from HDF5 file
        # extract parameters from ICESat2 HDF5 file
        if isinstance(self.filename, str):
            # extract parameters from ICESat2 HDF5 file
            SUB,PRD,HEM,YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,RL,VERS,AUX = \
                rx.findall(os.path.basename(self.filename)).pop()
        else:
            SUB,PRD,HEM,YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,RL,VERS,AUX = \
                rx.findall(os.path.basename(self.filename.filename)).pop()

        # copy bare minimum variables from the HDF5 file to pandas data frame
        source = h5py.File(self.filename,mode='r')

        # find valid beam groups by testing for particular variables
        if (PRD == 'ATL06'):
            VARIABLE_PATH = ['land_ice_segments','segment_id']
        elif (PRD == 'ATL07'):
            VARIABLE_PATH = ['sea_ice_segments','height_segment_id']
        elif (PRD == 'ATL08'):
            VARIABLE_PATH = ['land_segments','segment_id_beg']
        elif (PRD == 'ATL10'):
            VARIABLE_PATH = ['freeboard_beam_segments','delta_time']
        elif (PRD == 'ATL12'):
            VARIABLE_PATH = ['ssh_segments']['delta_time']
        # create list of valid beams within the HDF5 file
        beams = []
        for gtx in [k for k in source.keys() if bool(re.match(r'gt\d[lr]',k))]:
            # check if subsetted beam contains data
            try:
                source['/'.join([gtx,*VARIABLE_PATH])]
            except KeyError:
                pass
            else:
                beams.append(gtx)

        # for each valid beam within the HDF5 file
        frames = []
        for gtx in sorted(beams):
            # create a column stack of valid output segment values
            if (PRD == 'ATL06'):
                var = source[gtx]['land_ice_segments']
                valid, = np.nonzero(var['h_li'][:] != var['h_li'].fillvalue)
                # variables for the output dataframe
                vnames = ['segment_id','delta_time','latitude','longitude',
                    'h_li','h_li_sigma','atl06_quality_summary']
                data = {key:var[key][valid] for key in vnames}
                # copy filename parameters
                data['rgt'] = [TRK]*len(valid)
                data['cycle'] = [CYCL]*len(valid)
                # copy beam-level attributes
                attrs = ['groundtrack_id','atlas_spot_number','atlas_beam_type',
                    'sc_orientation','atmosphere_profile','atlas_pce']
                for att_name in attrs:
                    att_val=self.attributes_encoder(source[gtx].attrs[att_name])
                    data[att_name] = [att_val]*len(valid)
                # pandas dataframe from compiled dictionary
                frames.append(pandas.DataFrame.from_dict(data))
        # return the concatenated pandas dataframe
        return pandas.concat(frames)

    # PURPOSE: encoder for copying the file attributes
    def attributes_encoder(self, attr):
        """Custom encoder for copying file attributes in Python 3"""
        if isinstance(attr, (bytes, bytearray)):
            return attr.decode('utf-8')
        if isinstance(attr, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
            np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(attr)
        elif isinstance(attr, (np.float_, np.float16, np.float32, np.float64)):
            return float(attr)
        elif isinstance(attr, (np.ndarray)):
            if not isinstance(attr[0], (object)):
                return attr.tolist()
        elif isinstance(attr, (np.bool_)):
            return bool(attr)
        elif isinstance(attr, (np.void)):
            return None
        else:
            return attr
