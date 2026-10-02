use async_trait::async_trait;
use tokio::sync::mpsc::Sender;

pub mod grpc;
pub mod stratum;

use crate::pow::BlockSeed;
use crate::{Error, MinerManager};

#[async_trait(?Send)]
pub trait Client {
    fn add_devfund(&mut self, address: String, percent: u16);
    async fn register(&mut self) -> Result<(), Error>;
    async fn listen(&mut self, miner: &mut MinerManager) -> Result<(), Error>;
    fn get_block_channel(&self) -> Sender<BlockSeed>;
    /// Persist funds-critical client state (the escrow journal) before the process exits.
    fn flush_escrow_state(&mut self) -> Result<(), Error> {
        Ok(())
    }
}
