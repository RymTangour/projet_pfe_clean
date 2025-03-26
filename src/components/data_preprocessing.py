import os
from dataclasses import dataclass
import pandas as pd  
from box.exceptions import BoxValueError  
import yaml
from box import ConfigBox  
from pathlib import Path
import numpy as np
import seaborn as sns
import zipfile
from pathlib import Path
import cupy as cp
from concurrent.futures import ThreadPoolExecutor
from src import get_logger

logging=get_logger()
from src.config.configuration import TurboProcessorConfig
        

class TurboProcessor:
    def __init__(self, processsor_config : TurboProcessorConfig):
        self.processor_config=processsor_config
        os.makedirs(self.processor_config.output_dir, exist_ok=True)
        self.validate_input_directory()

        self.mem_pool=cp.cuda.MemoryPool()
        cp.cuda.set_allocator(self.mem_pool.malloc) 

        self.pinned_pool = cp.cuda.PinnedMemoryPool()
        cp.cuda.set_pinned_memory_allocator(self.pinned_pool.malloc)

        cp.fuse() 

        self.stream = cp.cuda.Stream(non_blocking=True)
        
        self.batch_size = 1024
        self.dtype = cp.float64
        self.define_kernels()

    def define_kernels(self):
        self.extract_lines_kernel = cp.ElementwiseKernel(
            'float64 start_time, float64 end_time, float64 time, float64 time_nano',
            'bool mask',
            'mask = (time * 1000000000 + time_nano >= start_time) && (time * 1000000000 + time_nano <= end_time)',
            'extract_lines_kernel'
        )
                    
        
              
           
            
        self.lba_variance_kernel = cp.ReductionKernel(
                'float64 lba, float64 mean',
                'float64 variance',
                '(lba - mean) * (lba - mean)',
                'a + b',
                'variance = a / _in_ind.size()',
                '0',
                'lba_variance_kernel'
            )
            
            
        self.entropy_kernel = cp.ElementwiseKernel(
                'float64 probability',
                'float64 entropy_term',
                'entropy_term = probability > 0 ? -probability * log2(probability) : 0',
                'entropy_kernel'
            )
            
        
        self.page_count_kernel = cp.ElementwiseKernel(
                'float64 page_type, float64 target_type',
                'int64 count',
                'count = (page_type == target_type) ? 1 : 0',
                'page_count_kernel'
            )
            
            
        self.gpa_variance_kernel = cp.ReductionKernel(
                'float64 gpa, float64 mean',
                'float64 variance',
                '(gpa - mean) * (gpa - mean)',
                'a + b',
                'variance = a / _in_ind.size()',
                '0',
                'gpa_variance_kernel'
            )


    def validate_input_directory(self):
        print(f'Checking input directory: {self.processor_config.input_dir}')
        print(f'Directory exists? {self.processor_config.input_dir.exists()}')
        print(f"Directory contents:{list(self.processor_config.input_dir.glob('*'))}")
        if not self.processor_config.input_dir.exists():
            raise FileNotFoundError(f"Container path missing: {self.processor_config.input_dir}")
        
    def get_batches(self):
        return [d for d in self.processor_config.input_dir.iterdir() if d.is_dir()]   
    
  
    def read_all_lines(self,data_gpu):
        return data_gpu
    
   
    def extract_lines(self, lines, t, T_window):
        if lines.size == 0:
            return lines

        t=self.dtype(t)  
        T_window=self.dtype(T_window)

        start_time = (lines[0, 0] + t) * (self.dtype(10)**9) + lines[0, 1]
        end_time = start_time + T_window * (self.dtype(10)**9)
        
        with self.stream:
            mask = self.extract_lines_kernel(start_time, end_time, lines[:, 0], lines[:, 1])
            result = lines[mask]
            
        return result
    
   
    def calc_throughput(self, lines):
        if lines.size == 0:
            return 0.0
        
        
        with self.stream:
            if lines.shape[0] >= 1:
                start_time_ns = lines[0, 0] * 10**9 + lines[0, 1]
                end_time_ns = lines[-1, 0] * 10**9 + lines[-1, 1]
                time_diff_ns = end_time_ns - start_time_ns
                
                if time_diff_ns <= 0:
                    return 0.0
                
                total_value = cp.sum(lines[:, 3])
                
                return float(total_value / time_diff_ns)
        
        return 0.0  

 
    

    def calc_LBA_variance(self, lines):
        if lines.size == 0:
            return 0.0
        with self.stream:
            lba_values = lines[:, 2]
            
            lba_mean = cp.mean(lba_values)
            
            variance = self.lba_variance_kernel(lba_values, lba_mean)
        return float(variance)
     
    def calc_Shannon_entropy(self,lines):
        if lines.size == 0:
            return 0.0
        
        with self.stream:
            col_values = lines[:, 4]
            unique_values, counts = cp.unique(col_values, return_counts=True)
            if len(unique_values) <= 1:
                return 0.0
                
            probabilities = counts.astype(cp.float64) / len(col_values)
            
            entropy_terms = self.entropy_kernel(probabilities)
            entropy = cp.sum(entropy_terms)
        self.stream.synchronize()    
        return float(entropy)
  
    def count_4KB_page_access(self,lines):
        if lines.size == 0:
            return 0
        
        with self.stream:
            page_types = lines[:, 5]
            
            counts = self.page_count_kernel(page_types, 1)
            total_count = cp.sum(counts)
            
        return int(total_count)
    
    def count_2MB_page_access(self,lines):
        if lines.size == 0:
            return 0
        
        with self.stream:
            page_types = lines[:, 5]
            
            counts = self.page_count_kernel(page_types, 2)
            total_count = cp.sum(counts)
            
        return int(total_count)
    def count_MMIO_page_access(self,lines):
        if lines.size == 0:
            return 0
        
        with self.stream:
            page_types = lines[:, 5]
            
            counts = self.page_count_kernel(page_types, 4)
            total_count = cp.sum(counts)
            
        return int(total_count)
    
   
    def calc_GPA_variance(self,lines):
        if lines.size == 0:
            return 0.0
        
        with self.stream:
            gpa_values = lines[:, 2]
            
            gpa_mean = cp.mean(gpa_values)
            
            variance = self.gpa_variance_kernel(gpa_values, gpa_mean)
            
        return float(variance)
    def calculate_storage_feature(self, ata_read_gpu, ata_write_gpu, T_d, T_window, eps):
        # Add debug logging
        logging.debug(f"Storage data shapes - Read: {ata_read_gpu.shape}, Write: {ata_write_gpu.shape}")
        
        t = 0
        XS = []
        
        lines_SR = self.read_all_lines(ata_read_gpu)
        lines_SW = self.read_all_lines(ata_write_gpu)

        if lines_SR.size == 0 or lines_SW.size == 0:
            return cp.empty((0, 5), dtype=self.dtype)



        steps = int((T_d - T_window) // eps) + 1
        XS = cp.zeros((steps, 5), dtype=self.dtype)


        with self.stream:
            step_idx = 0
            t = 0
        
        while t <= (T_d - T_window):
            lines_SRT =self. extract_lines(lines_SR, t, T_window)
            lines_SWT = self.extract_lines(lines_SW, t, T_window)
            T_SR =self.calc_throughput(lines_SRT)
            T_SW = self.calc_throughput(lines_SWT)
            V_SR = self.calc_LBA_variance(lines_SRT)
            V_SW = self.calc_LBA_variance(lines_SWT)
            H_SW = self.calc_Shannon_entropy(lines_SWT)
            XS[step_idx, 0] = T_SR
            XS[step_idx, 1] = T_SW
            XS[step_idx, 2] = V_SR
            XS[step_idx, 3] = V_SW
            XS[step_idx, 4] = H_SW
                
            step_idx += 1
            t += eps

            if step_idx % self.batch_size == 0:
                self.stream.synchronize()

        if step_idx < steps:
            XS = XS[:step_idx]
            
        return XS
    
    def calculate_memory_feature(self, mem_read_gpu, mem_write_gpu, mem_rw_gpu, mem_exec_gpu, 
                            T_d, T_window, eps):
        # Add debug logging
        logging.debug(f"Memory data shapes - Read: {mem_read_gpu.shape}, Write: {mem_write_gpu.shape}")
        logging.debug(f"Memory data shapes - RW: {mem_rw_gpu.shape}, Exec: {mem_exec_gpu.shape}")
        
        t = 0
        XM = []
        lines_MR = self.read_all_lines(mem_read_gpu)
        lines_MW = self.read_all_lines(mem_write_gpu)
        lines_MRW = self.read_all_lines(mem_rw_gpu)
        lines_MX = self.read_all_lines(mem_exec_gpu)

        if (lines_MR.size == 0 or lines_MW.size == 0 or 
        lines_MRW.size == 0 or lines_MX.size == 0):
        
            return cp.empty((0, 18), dtype=self.dtype)

        steps = int((T_d - T_window) // eps) + 1
        XM = cp.zeros((steps, 18), dtype=self.dtype)
        with self.stream:
            step_idx = 0
            t = 0

        while t <= (T_d - T_window):
            lines_MRT = self.extract_lines(lines_MR, t, T_window)
            lines_MWT = self.extract_lines(lines_MW, t, T_window)
            lines_MRWT = self.extract_lines(lines_MRW, t, T_window)
            lines_MXT = self.extract_lines(lines_MX, t, T_window)
                
            H_MW = self.calc_Shannon_entropy(lines_MWT)
            H_MRW = self.calc_Shannon_entropy(lines_MRWT)  

            C_4KB_R = self.count_4KB_page_access(lines_MRT)
            C_4KB_W = self.count_4KB_page_access(lines_MWT)
            C_4KB_RW = self.count_4KB_page_access(lines_MRWT)
            C_4KB_X = self.count_4KB_page_access(lines_MXT)

            C_2MB_R = self.count_2MB_page_access(lines_MRT)
            C_2MB_W = self.count_2MB_page_access(lines_MWT)
            C_2MB_RW = self.count_2MB_page_access(lines_MRWT)
            C_2MB_X = self.count_2MB_page_access(lines_MXT)

            
            C_MMIO_R = self.count_MMIO_page_access(lines_MRT)
            C_MMIO_W = self.count_MMIO_page_access(lines_MWT)  
            C_MMIO_RW = self.count_MMIO_page_access(lines_MRWT)
            C_MMIO_X = self.count_MMIO_page_access(lines_MXT)

            V_MR = self.calc_GPA_variance(lines_MRT)
            V_MW = self.calc_GPA_variance(lines_MWT)
            V_MRW = self.calc_GPA_variance(lines_MRWT)
            V_MX = self.calc_GPA_variance(lines_MXT)
            
            XM[step_idx, 0] = H_MW
            XM[step_idx, 1] = H_MRW
            XM[step_idx, 2] = C_4KB_R
            XM[step_idx, 3] = C_4KB_W
            XM[step_idx, 4] = C_4KB_RW
            XM[step_idx, 5] = C_4KB_X
            XM[step_idx, 6] = C_2MB_R
            XM[step_idx, 7] = C_2MB_W
            XM[step_idx, 8] = C_2MB_RW
            XM[step_idx, 9] = C_2MB_X
            XM[step_idx, 10] = C_MMIO_R
            XM[step_idx, 11] = C_MMIO_W
            XM[step_idx, 12] = C_MMIO_RW
            XM[step_idx, 13] = C_MMIO_X
            XM[step_idx, 14] = V_MR
            XM[step_idx, 15] = V_MW
            XM[step_idx, 16] = V_MRW
            XM[step_idx, 17] = V_MX
            step_idx += 1
            t += eps
            if step_idx % self.batch_size == 0:
                self.stream.synchronize()
        if step_idx < steps:
            XM = XM[:step_idx]
            
        return XM
    def process_file_group(self, file_group, output_path):

        try:
            with cp.cuda.Stream() as stream:
                storage_data = {}
                memory_data = {}
                logging.info(f"Loading data files for {output_path}")
                with ThreadPoolExecutor(max_workers=2) as executor:

                    storage_future = executor.submit(
                        lambda: {
                            'ata_read': cp.array(pd.read_csv(file_group['ata_read'], header=None).values, dtype=self.dtype),
                            'ata_write': cp.array(pd.read_csv(file_group['ata_write'], header=None).values, dtype=self.dtype)
                        }
                    )
                    
                    memory_future = executor.submit(
                        lambda: {
                            'mem_read': cp.array(pd.read_csv(file_group['mem_read'], header=None).values, dtype=self.dtype),
                            'mem_write': cp.array(pd.read_csv(file_group['mem_write'], header=None).values, dtype=self.dtype),
                            'mem_rw': cp.array(pd.read_csv(file_group['mem_readwrite'], header=None).values, dtype=self.dtype),
                            'mem_exec': cp.array(pd.read_csv(file_group['mem_exec'], header=None).values, dtype=self.dtype),
                        }
                    )
                    storage_data = storage_future.result()
                    memory_data = memory_future.result()

                storage_features=self.calculate_storage_feature(
                    storage_data['ata_read'],
                    storage_data['ata_write'],
                    self.processor_config.T_d,
                    self.processor_config.T_window,
                    self.processor_config.eps  
                )

                memory_features=self.calculate_memory_feature(
                    memory_data['mem_read'],
                    memory_data['mem_write'],
                    memory_data['mem_rw'],
                    memory_data['mem_exec'],
                    self.processor_config.T_d,
                    self.processor_config.T_window,
                    self.processor_config.eps
                )

                stream.synchronize()

                if (storage_features.size == 0)or  (memory_features.size == 0):
                    logging.warning(f"No features generated for {output_path}")
                    return False
                
                os.makedirs(output_path.parent, exist_ok=True)

                storage_path=output_path.parent / 'storage.npy'
                memory_path=output_path.parent / 'memory.npy'

                with self.stream:
                    cp.save(storage_path, storage_features)
                with self.stream:
                    cp.save(memory_path, memory_features)

                self.stream.synchronize()
                logging.info(f"Saved features to {output_path.parent}")
                return True
            
        except Exception as e:
            logging.error(f"Error processing file group: {str(e)}")
            return False
        finally:
            self.mem_pool.free_all_blocks()
            self.pinned_pool.free_all_blocks()

    def process_batch(self, batch_dir):
        try:
            logging.info(f"\n{'='*40}")
            logging.info(f"STARTING BATCH: {batch_dir.name}")
            logging.info(f"\n{'='*40}")

            if not batch_dir.exists() or not batch_dir.is_dir():
                logging.error(f"Batch directory does not exist or is not a directory: {batch_dir}")
                return False
            
            processed_count = 0 
            required_files = {
                'ata_read': 'ata_read.csv',
                'ata_write': 'ata_write.csv',
                'mem_read': 'mem_read.csv',
                'mem_write': 'mem_write.csv',
                'mem_readwrite': 'mem_readwrite.csv',
                'mem_exec': 'mem_exec.csv'
            }

            ram_type_dirs = [d for d in batch_dir.iterdir() if d.is_dir()]
            
            max_concurrent = min(os.cpu_count(), 4)
            futures = []
            
            with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                for ram_type_dir in ram_type_dirs:
                    program_dirs = [d for d in ram_type_dir.iterdir() if d.is_dir()]
                    
                    for program_dir in program_dirs:
                        timestamp_dirs = [d for d in program_dir.iterdir() if d.is_dir()]
                        
                        for timestamp_dir in timestamp_dirs:
                            file_group = {}
                            missing_files = False
                            
                            for key, fname in required_files.items():
                                path = timestamp_dir / fname
                                if not path.exists():
                                    logging.warning(f"Missing required file: {path}")
                                    missing_files = True
                                    break
                                
                                file_group[key] = path
                            
                            if missing_files:
                                continue
                            
                            relative_path = timestamp_dir.relative_to(self.processor_config.input_dir)
                            output_path = self.processor_config.output_dir / relative_path / "features.npy"
                            
                            future = executor.submit(self.process_file_group, file_group, output_path)
                            futures.append(future)
                
                for future in futures:
                    try:
                        if future.result(timeout=1800):  
                            processed_count += 1
                    except Exception as e:
                        logging.error(f"File group processing failed: {str(e)}")
                    finally:
                        self.mem_pool.free_all_blocks()
                        self.pinned_pool.free_all_blocks()
            
            total_dirs = sum(len([d for d in ram_dir.iterdir() if d.is_dir()]) for ram_dir in ram_type_dirs 
                            for program_dir in [d for d in ram_dir.iterdir() if d.is_dir()])
            
            logging.info(f"Batch {batch_dir.name}: {processed_count}/{total_dirs} groups processed")
            return True
        
        except Exception as e:
            logging.error(f"Batch failed: {str(e)}")
            return False

    def run(self):
        
        batches = self.get_batches()
        if not batches:
            logging.warning("No batches found to process")
            return
            
        logging.info(f"Found {len(batches)} batches to process")
        
        
        for batch in batches:
            try:
                logging.info(f"Processing batch: {batch.name}")
                self.process_batch(batch)
            except Exception as e:
                logging.error(f"Error processing batch {batch.name}: {str(e)}")
            finally:
                self.mem_pool.free_all_blocks()
                self.pinned_pool.free_all_blocks()
                cp.get_default_memory_pool().free_all_blocks()
                
        logging.info("All batches processed")





