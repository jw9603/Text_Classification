import torch
import torch.nn.utils as torch_utils

from ignite.engine import Events

from my_ntc.utils import get_grad_norm, get_parameter_norm

VERBOSE_SLIENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2

from my_ntc.trainer import Trainer, MyEngine

class EngineForBERT(MyEngine):
    
    def __init__(self,func,model,crit,optimizer,scheduler,config):
        self.scheduler = scheduler
        
        super().__init__(func,model,crit,optimizer,config)
        
    @staticmethod # 클래스에서 인스턴스를 부르지 않고 바로 메서드를 부를 수 있음
    def train(engine,mini_batch):
        # you have to reset the gradients of all model parameters
        # before to take another step in gradient descent.
        engine.model.train()
        engine.optimizer.zero_grad()
        
        x, y = mini_batch['input_ids'], mini_batch['labels']
        x, y = x.to(engine.device), y.to(engine.device)
        mask = mini_batch['attention_mask']
        mask = mask.to(engine.device)
        
        x = x[:,:engine.config.max_length] # max length로 slicing
        
        # Take feed-forward
        y_hat = engine.model(x,attention_mask=mask).logits 
        # |y_hat| = (n,|c|), n=mini_batch
        
        loss = engine.crit(y_hat,y)
        loss.backward()
        
        if isinstance(y,torch.LongTensor) or isinstance(y,torch.cuda.LongTensor):
            acc = (torch.argmax(y_hat,dim=-1)==y).sum() / float(y.size(0))
        else:
            acc = 0
        
        p_norm = float(get_parameter_norm(engine.model.parameters()))
        g_norm = float(get_grad_norm(engine.model.parameters()))

        # Take a step of gradient descent.
        engine.optimizer.step()
        engine.scheduler.step()

        return {
            'loss': float(loss),
            'accuracy': float(acc),
            '|param|': p_norm,
            '|g_param|': g_norm,
        }


class BERTTrainer(Trainer):
    
    def __init__(self,config):
        self.config = config
        
    def train(self,model,crit,optimizer,scheduler,train_loader,valid_loader):
        
        train_engine = EngineForBERT(
            EngineForBERT.train, # process function
            model, crit, optimizer, scheduler, self.config
        )
        
        validation_engine = EngineForBERT(
            EngineForBERT.validate,
            model,crit, optimizer,scheduler,self.config
        )
        
        EngineForBERT.attach(
            train_engine,
            validation_engine,
            verbose=self.config.verbose
        )
        
        def run_validate(engine,validation_engine,valid_loader):
        # run : Runs the process_function over the passed data.
        # run(data=None, max_epochs=None, epoch_length=None, seed=None)
            # parameters
                # data (Optional[Iterable]) – Collection of batches allowing repeated iteration (e.g., list or DataLoader). If not provided, then epoch_length is required and batch argument of process_function will be None.
                # max_epochs (Optional[int]) – Max epochs to run for (default: None). If a new state should be created (first run or run again from ended engine), it’s default value is 1. If run is resuming from a state, provided max_epochs will be taken into account and should be larger than engine.state.max_epochs.
                # epoch_length (Optional[int]) – Number of iterations to count as one epoch. By default, it can be set as len(data). If data is an iterator and epoch_length is not set, then it will be automatically determined as the iteration on which data iterator raises StopIteration. This argument should not change if run is resuming from a state.
                # seed (Optional[int]) – Deprecated argument. Please, use torch.manual_seed or manual_seed().
            validation_engine.run(valid_loader,max_epochs=1)
        
        # https://pytorch.org/ignite/generated/ignite.engine.events.Events.html
        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED, # event
            run_validate,            # It is call-back function
            validation_engine,valid_loader                    
        )
        # Since it is a callback function, the arguments of the callback function can be called as the arguments of other functions.
    
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED,     # event
            EngineForBERT.check_best    # function
        )
    
        train_engine.run(train_loader,max_epochs=self.config.n_epochs)
    
        model.load_state_dict(validation_engine.best_model)
        
        return model