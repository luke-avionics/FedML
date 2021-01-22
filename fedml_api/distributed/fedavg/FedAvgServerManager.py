import logging
import copy

from fedml_api.distributed.fedavg.message_define import MyMessage
from fedml_api.distributed.fedavg.utils import transform_tensor_to_list
from fedml_core.distributed.communication.message import Message
from fedml_core.distributed.server.server_manager import ServerManager
from fedml_api.model.cv.quantize import calculate_qparams, quantize
from fedml_api.distributed.fedavg.FedAVGTrainer import ServerTrainer

class FedAVGServerManager(ServerManager):
    def __init__(self, args, aggregator, comm=None, rank=0, size=0, backend="MPI", device='cuda:0', use_fake_data=True):
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.round_idx = 0
        self.traffic_count=0
        self.client_indexes=[]

        self.device = device
        self.use_fake_data = use_fake_data

        if use_fake_data:
            self.server_trainer = ServerTrainer(device=device) # currently model is fixed
        else:
            self.server_trainer = None

    def run(self):
        super().run()

    def send_init_msg(self):
        # sampling clients
        self.client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                                                         self.args.client_num_per_round)
        global_model_params = self.aggregator.get_global_model_params()
        for process_id in range(1, self.size):
            self.send_message_init_config(process_id, global_model_params, self.client_indexes[process_id-1])

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
                                              self.handle_message_receive_model_from_client)

    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)

        try:
            for received_pack in model_params.keys():
                tmp_traffic=1
                for tmp_dim in model_params[received_pack].shape:
                    tmp_traffic*=tmp_dim
                    self.traffic_count+=int(tmp_traffic)

            # logging.info("Traffic consummed: "+str(self.traffic_count))
            #wandb.log({"Traffic consummed": self.traffic_count, "mini_round": self.round_idx},commit=False)
        except Exception as e:
            logging.info(str(e))
        self.aggregator.add_local_trained_result(sender_id - 1, model_params, local_sample_number)
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.info("b_all_received = " + str(b_all_received))

        if b_all_received:
            global_model_params = self.aggregator.aggregate()
            try:
                self.aggregator.test_on_all_clients(self.round_idx,self.traffic_count,self.client_indexes)
            except Exception as e:
                raise Exception(str(e))
            # start the next round
            self.round_idx += 1
            if self.round_idx == self.round_num:
                self.finish()
                return

            # sampling clients
            self.client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                                                             self.args.client_num_per_round)
            print("size = %d" % self.size)
            # if self.args.is_mobile == 1:
            #     print("transform_tensor_to_list")
            #     global_model_params = transform_tensor_to_list(global_model_params)

            if self.round_idx <= 1 or self.use_fake_data == False:
                shared_data = None
            else:
                shared_data = self.shared_data

            for receiver_id in range(1, self.size):
                #self.send_message_sync_model_to_client(receiver_id, self.aggregator.model_dict[receiver_id-1], self.client_indexes[receiver_id-1])
                self.send_message_sync_model_to_client(receiver_id, global_model_params, self.client_indexes[receiver_id-1], shared_data)
            # logging.info("++++++++++++++reaching here")
            # logging.info(str(self.use_fake_data))
            # logging.info(str(self.server_trainer))
            if self.use_fake_data and self.server_trainer is not None:
                self.shared_data = self.server_trainer.generate_fake_data(copy.deepcopy(global_model_params))
                logging.info('Finished generating fake data..............')


    def send_message_init_config(self, receive_id, global_model_params, client_index):
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)


    def send_message_sync_model_to_client(self, receive_id, global_model_params, client_index, shared_data=None):
        logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))

        message.add_params(MyMessage.MSG_ARG_KEY_SHARE_DATA, shared_data)

        self.send_message(message)
