import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

CONTINUOUS = ("log_bytes","log_packets","log_duration","iat_mean","iat_std","iat_cv","bytes_per_packet","bytes_per_sec","packets_per_sec","payload_entropy","flag_count")
BINARY = ("flag_syn","flag_ack","flag_fin","flag_rst","flag_psh","flag_urg","is_syn_only","is_encrypted_port","proto_tcp","proto_udp","proto_icmp")
CATEGORICAL = ("src_port_category","dst_port_category","tos")

def numeric(batch, name):
    idx = batch.schema.get_field_index(name)
    if idx < 0:
        raise ValueError(f"Missing feature: {name}")
    return pc.cast(batch.column(idx), pa.float64()).to_numpy(zero_copy_only=False)

def port_category(x):
    return np.select([np.isfinite(x)&(x>=0)&(x<=1023)&(x==np.floor(x)), np.isfinite(x)&(x>=1024)&(x<=49151)&(x==np.floor(x)), np.isfinite(x)&(x>=49152)&(x<=65535)&(x==np.floor(x))], [1,2,3], default=0).astype(np.int16)

def extract_features(batch):
    continuous = np.column_stack([numeric(batch,n) for n in CONTINUOUS])
    binary = np.column_stack([numeric(batch,n) for n in BINARY])
    if not np.isin(binary,[0,1]).all():
        raise ValueError("Invalid or missing binary feature")
    tos = numeric(batch,"tos")
    tos = np.where(np.isfinite(tos)&(tos>=0)&(tos<=255)&(tos==np.floor(tos)),tos,-1)
    categorical = np.column_stack([port_category(numeric(batch,"src_port")),port_category(numeric(batch,"dst_port")),tos]).astype(np.int16)
    return continuous, binary.astype(np.uint8), categorical
